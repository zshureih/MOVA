import os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from args import parse_args
from media import get_lyrics, get_audio

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import PIL
import moviepy.editor as mpe
from image_bind.models.imagebind_model import ModalityType
import image_bind.data as ibd

torch.set_grad_enabled(False)


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    print("Model instantiated")
    m, u = model.load_state_dict(sd, strict=False)
    print("Model state dict loaded")
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model

def interpolate(init, end, t, interpolation):
    """Interpolates between two prompts using the specified method.
    
    Args:
        init (Tensor): The encoded initial prompt.
        end (Tensor): The encoded final prompt.
        t (float): The interpolation factor.
        interpolation (str): The interpolation method.
        
    Returns:
        Tensor: The interpolated prompt.    
    """
    if interpolation == "linear":
        interpolated = (1 - t) * init + t * end
    elif interpolation == "cosine":
        t_cosine = (1 - torch.cos(t * torch.pi)) / 2
        interpolated = (1 - t_cosine) * init + t_cosine * end
    elif interpolation == "cubic":
        t_cubic = t**3 * (t * (6 * t - 15) + 10)
        interpolated = (1 - t_cubic) * init + t_cubic * end
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation}")

    return interpolated

def main_bind(opt):
    seed_everything(opt.seed)

    root_dir = Path(__file__).resolve().parent.parent.parent
    print(f"Root directory: {root_dir}")

    pathsep = "/" if os.name != "nt" else "\\"
    
    config_path = opt.config.split(pathsep)
    config_path = root_dir.joinpath(*config_path)
    
    ckpt_path = opt.ckpt.split(pathsep)
    ckpt_path = root_dir.joinpath(*ckpt_path)

    config = OmegaConf.load(f"{config_path}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{ckpt_path}", device)

    print("Loaded Model, setting up Sampler")

    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)
    print("Sampler setup complete")

    outpath = root_dir / opt.outdir
    os.makedirs(outpath, exist_ok=True)

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "SDV2"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = 1
    negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    # get information about the song
    artist, title = opt.artist, opt.song
    lyrics, audio_path = None, None
    if artist is not None and title is not None:
        print(f"Generating samples for {artist} - {title}")
        # get the song's lyrics
        lyrics = get_lyrics(artist, title)
        # get the song's audio
        if not opt.init_audio:
            audio_path = get_audio(artist, title)
        else:
            audio_path = opt.init_audio
    else:
        print("No artist or title provided, try again.")
        return 0

    # setup the prompts
    text_prompts = []
    if lyrics is not None:
        # generate prompts from the lyrics
        for section in lyrics:
            prompt = opt.prompt + f", set to music with the lyrics: {section}"
            prompt += f", best quality, extremely detailed"
            text_prompts.append(prompt)
    else:
        # we couldn't find any lyrics, so we'll return false
        print("No lyrics found.")
        return 0

    # setup output directory
    sample_path = os.path.join(outpath, f"{artist}_{title}_{opt.interpolation}_samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))

    precision_scope = autocast if opt.precision == "autocast" or opt.bf16 else nullcontext

    # setup text_prompts for interpolation
    start_codes = [torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device) for _ in range(len(text_prompts))]

    start_codes = [(start_codes[i], start_codes[i + 1]) for i in range(0, len(start_codes) - 1)]
    text_prompts = [(text_prompts[i], text_prompts[i + 1]) for i in range(0, len(text_prompts) - 1)]

    duration = 30 # seconds
    interpolation_steps = (opt.fps * duration) // len(text_prompts)

    # start generating samples
    with torch.no_grad(), precision_scope(opt.device), model.ema_scope():
        frames = list()

        # encode the audio sample
        init_audio = ibd.load_and_transform_audio_data([audio_path], device)
        inputs = {
            ModalityType.AUDIO: init_audio,
        }
        outs = model.embedder(inputs, normalize=False)[ModalityType.AUDIO]
        c_adm = repeat(outs, '1 ... -> b ...', b=batch_size) * opt.strength

        # add noise to the conditioning
        if model.noise_augmentor is not None:
            c_adm, noise_level_emb = model.noise_augmentor(
                c_adm,
                noise_level=(
                    torch.ones(batch_size)
                    * model.noise_augmentor.max_noise_level
                    * opt.noise_level
                )
                .long()
                .to(c_adm.device),
            )
            # assume this gives embeddings of noise levels
            c_adm = torch.cat((c_adm, noise_level_emb), 1)

        # add unconditional conditioning
        uc = model.get_learned_conditioning([negative_prompt])
        uc = {"c_crossattn": [uc], "c_adm": c_adm}

        # iterate over the text prompts
        for i, prompts in enumerate(tqdm(text_prompts)):
            # encode the initial and next prompts
            initial_prompt = model.get_learned_conditioning(prompts[0])
            next_prompt = model.get_learned_conditioning(prompts[1])

            initial_code = start_codes[i][0]
            next_code = start_codes[i][1]

            for t in torch.linspace(0, 1, interpolation_steps):
                # interpolate between the prompts
                interpolated = interpolate(initial_prompt, next_prompt, t, opt.interpolation)
                interpolated_code = interpolate(
                    initial_code, next_code, t, opt.interpolation
                )

                c = {
                    "c_crossattn": [interpolated],
                    "c_adm": c_adm,
                }
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples, _ = sampler.sample(
                    S=opt.sample_steps,
                    conditioning=c,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta,
                    x_T=interpolated_code,
                )

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                x_sample = 255. * rearrange(x_samples[-1].cpu().numpy(), 'c h w -> h w c')

                img = Image.fromarray(x_sample.astype(np.uint8))
                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1
                sample_count += 1

                frames.append(img)

    # save frames as a video @ 30fps
    video_path = os.path.join(
        outpath, f"{artist}_{title}_{opt.interpolation}_video.mp4"
    )
    frame_height, frame_width = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, opt.fps, (frame_width, frame_height))

    for frame in frames:
        video_writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

    video_writer.release()

    # Combine video with audio
    video_clip = mpe.VideoFileClip(video_path)
    audio_clip = mpe.AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(
        os.path.join(outpath, f"{artist}_{title}_{opt.interpolation}_final_video.mp4"),
        codec="libx264",
    )

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

    return 1

if __name__ == "__main__":
    opt = parse_args()
    # print(opt)
    main_bind(opt)
