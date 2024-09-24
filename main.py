import os
import torch
import argparse
import torch.optim as optim
import torch.distributed as dist
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from train import train_vdt
from eval import test_vdt
from models import VDT_models
from preprocess import FrameDataset
from diffusion import create_diffusion
from metrics import MetricCalculator
from diffusers.models import AutoencoderKL
from utils import setup_logging, ddp_setup, load_checkpoint

def main_paral(args=None):
    ddp_setup()
    device = int(os.environ["LOCAL_RANK"])
    model = VDT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        mode="video",
        cfg_scale=args.cfg_scale,
    )

    if args.ckpt is not None:
        model, _ = load_checkpoint(model, args.ckpt)

    model = model.to(device)
    model = DDP(model, device_ids=[device])
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps), training=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    metrics_calculator = MetricCalculator(["SSIM", "PSNR", "LPIPS", "FVD"], device=device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = FrameDataset(root_dir="leftImg8bit_sequence_preprocess/train", transform=transform, frames_per_clip=args.num_frames)
    val_dataset = FrameDataset(root_dir="leftImg8bit_sequence_preprocess/val", transform=transform, frames_per_clip=args.num_frames)
    test_dataset = FrameDataset(root_dir="leftImg8bit_sequence_preprocess/test", transform=transform, frames_per_clip=args.num_frames)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler)

    if args.run_mode == "train":
        train_vdt(args, model, train_dataloader, val_dataloader, vae, diffusion, optimizer, device, metrics_calculator, num_epochs=args.epoch)
    
    test_vdt(args, model, test_dataloader, vae, diffusion, device, metrics_calculator)
    
    dist.destroy_process_group()

def main_single(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VDT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        mode="video",
        cfg_scale=args.cfg_scale,
    )
    if args.ckpt is not None:
        model, _ = load_checkpoint(model, args.ckpt)
    model = model.to(device)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps), training=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    metrics_calculator = MetricCalculator(["SSIM", "PSNR", "LPIPS", "FVD"], device=device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = FrameDataset(root_dir="leftImg8bit_sequence_preprocess/train", transform=transform)
    val_dataset = FrameDataset(root_dir="leftImg8bit_sequence_preprocess/val", transform=transform)
    test_dataset = FrameDataset(root_dir="leftImg8bit_sequence_preprocess/test", transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.run_mode == "train":
        train_vdt(args, model, train_dataloader, val_dataloader, vae, diffusion, optimizer, device, metrics_calculator, num_epochs=args.epoch, cfg_scale=1.0)

    test_vdt(args, model, test_dataloader, vae, diffusion, device, metrics_calculator)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-L/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--f", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default="vdt_model.pt", help="Optional path to a VDT checkpoint.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--mode", type=str, choices=["single", "paral"], default="single")
    parser.add_argument("--run_mode", type=str, choices=["train", "test"], default="train")
    args = parser.parse_args()
    
    setup_logging()
    
    if args.mode == "paral":
        main_paral(args=args)
    elif args.mode == "single":
        main_single(args=args)
    else:
        raise("There is no such mode!")