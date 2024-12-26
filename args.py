import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--song",
        type=str,
        default="GOMD",
        help="the title of the song to generate lyrics for",
    )
    parser.add_argument(
        "--artist",
        type=str,
        default="J. Cole",
        help="the artist of the selected song",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="wide angle, artistic shots from a music video",
        help="the prompt to add to the lyrics",
    )
    parser.add_argument(
        "--init-audio",
        type=str,
        default=None,
        nargs="?",
        help="path to the input modality",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs",
    )
    parser.add_argument(
        "--isteps",
        type=int,
        default=900,
        help="number of interpolation steps",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0,
        help="noise level",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="strength",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="mix rate",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action="store_true",
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-1-stable-unclip-h-bind-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/sd21-unclip-h.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=316199,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--torchscript",
        action="store_true",
        help="Use TorchScript",
    )
    parser.add_argument(
        "--ipex",
        action="store_true",
        help="Use Intel® Extension for PyTorch*",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        help="interpolation mode",
        choices=["linear", "cosine", "cubic"],
        default="cosine",
    )
    opt = parser.parse_args()
    return opt
