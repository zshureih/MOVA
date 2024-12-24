import os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
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
from image_bind.models.imagebind_model import ModalityType

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


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def main(opt):
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device)

    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "SDV2"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    # get information about the song
    artist, title = opt.artist, opt.song
    if artist is not None and title is not None:
        print(f"Generating samples for {artist} - {title}")
        # get the song's lyrics
        lyrics = get_lyrics(artist, title)
        # get the song's audio
        audio = get_audio(artist, title)
    

    # if reading from command line
    if not opt.from_file:
        prompt = opt.prompt + ', best quality, extremely detailed'
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(opt.repeat)]
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision == "autocast" or opt.bf16 else nullcontext
    with torch.no_grad(), \
            precision_scope(opt.device), \
            model.ema_scope():
        all_samples = list()
        for n in trange(opt.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                import image_bind.data as ibd
                if opt.modality == 'audio':
                    init_audio = ibd.load_and_transform_audio_data([opt.init], device)
                    inputs = {
                        ModalityType.AUDIO: init_audio,
                    }
                    embeddings = model.embedder(inputs, normalize=True)[ModalityType.AUDIO]
                elif opt.modality == 'image':
                    init_image = ibd.load_and_transform_vision_data([opt.init], device)
                    inputs = {
                        ModalityType.VISION: init_image,
                    }
                    embeddings = model.embedder(inputs, normalize=False)[ModalityType.VISION]
                else:
                    assert False
                c_adm = repeat(embeddings, '1 ... -> b ...', b=batch_size) * opt.strength

                if model.noise_augmentor is not None:
                    c_adm, noise_level_emb = model.noise_augmentor(c_adm, noise_level=(
                            torch.ones(batch_size) * model.noise_augmentor.max_noise_level *
                            opt.noise_level).long().to(c_adm.device))
                    # assume this gives embeddings of noise levels
                    c_adm = torch.cat((c_adm, noise_level_emb), 1)
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [negative_prompt])
                if opt.uc_zero:
                    uc = {"c_crossattn": [uc], "c_adm": torch.zeros_like(c_adm)}
                else:
                    uc = {"c_crossattn": [uc], "c_adm": c_adm}
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = {"c_crossattn": [model.get_learned_conditioning(prompts)], "c_adm": c_adm}
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples, _ = sampler.sample(S=opt.steps,
                                            conditioning=c,
                                            batch_size=opt.n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc,
                                            eta=opt.ddim_eta,
                                            x_T=start_code)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    # img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1
                    sample_count += 1

                all_samples.append(x_samples)

        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=n_rows)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))
        # grid = put_watermark(grid, wm_encoder)
        grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
        grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
