import os
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from diffusers.models import AutoencoderKL

from models import VDT_models
from preprocess import FrameDataset
from diffusion import create_diffusion
from mask_generator import VideoMaskGenerator

from metrics import MetricCalculator

import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def ddp_setup():
    dist.init_process_group(backend="nccl")

def train_vdt(model, train_dataloader, val_dataloader, 
              vae, diffusion, optimizer, device, num_epochs=10, 
              cfg_scale=1.0):
    model.to(device)
    metrics_calculator = MetricCalculator(['SSIM', 'PSNR', 'LPIPS', 'FVD'], device=device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, x in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            B, T, C, H, W = x.shape
            x = x.to(device)

            with torch.no_grad():
                latent_x = vae.encode(x.view(-1, C, H, W)).latent_dist.sample().mul_(0.18215)
            
            latent_x = latent_x.view(B, T, -1, latent_x.shape[-2], latent_x.shape[-1])
            
            choice_idx = random.randint(0, 6)
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]))
            mask = generator(B, device, idx=choice_idx)
            
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()
            latent_x = latent_x.to(device)

            sample_fn = model.forward
            results = diffusion.training_losses(sample_fn, latent_x, t, 
                                                  noise=None, mask=mask)
            
            loss = results["loss"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                samples = results['output'].clone()
                samples = samples.permute(1, 0, 2, 3, 4) * mask + latent_x.permute(2, 0, 1, 3, 4) * (1-mask)
                samples = samples.permute(1, 2, 0, 3, 4).reshape(-1, 4, latent_x.shape[-2], latent_x.shape[-1]) / 0.18215
                decoded_samples = decode_in_batches(samples, vae, chunk_size=256)
                decoded_samples = decoded_samples.view(B, T, decoded_samples.shape[-3], decoded_samples.shape[-2], decoded_samples.shape[-1])
                metrics = metrics_calculator(x.to('cpu'), decoded_samples.to('cpu'))
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], "
                            f"Loss: {running_loss / num_epochs:.4f}, "
                            f"SSIM: {metrics['SSIM'].mean():.4f}, "
                            f"PSNR: {metrics['PSNR'].mean():.4f}, "
                            f"LPIPS: {metrics['LPIPS'].mean():.4f}, "
                            f"FVD: {metrics['FVD']:.4f}, ")

                running_loss = 0.0
        
        validate_vdt(model, val_dataloader, vae, diffusion, device, metrics_calculator)
        diffusion.training = True
    torch.save(model.state_dict(), 'vdt_model.pt')
    print("Training finished.")

def validate_vdt(model, val_dataloader, vae, diffusion, device, metrics_calculator):
    model.eval()
    running_loss = 0.0
    criterion = nn.MSELoss()

    ssim_scores  = []
    psnr_scores  = []
    lpips_scores = []
    fvd_scores   = []
    diffusion.training = False
    with torch.no_grad():
        for batch_idx, x in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            B, T, C, H, W = x.shape
            x = x.to(device)
            
            with torch.no_grad():
                latent_x = vae.encode(x.view(-1, C, H, W)).latent_dist.sample().mul_(0.18215)
            latent_x = latent_x.view(B, T, -1, latent_x.shape[-2], latent_x.shape[-1])
            
            choice_idx = random.randint(0, 6)
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]))
            mask = generator(B, device, idx=choice_idx)
            
            z = torch.randn(B, T, 4, latent_x.shape[-2], latent_x.shape[-1], device=device).permute(0, 2, 1, 3, 4)
            latent_x = latent_x.to(device)

            sample_fn = model.forward
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, progress=False, device=device,
                raw_x=latent_x, mask=mask
            )
            samples = samples.permute(1, 0, 2, 3, 4) * mask + latent_x.permute(2, 0, 1, 3, 4) * (1-mask)
            samples = samples.permute(1, 2, 0, 3, 4).reshape(-1, 4, latent_x.shape[-2], latent_x.shape[-1]) / 0.18215
            
            decoded_samples = decode_in_batches(samples, vae, chunk_size=256)
            decoded_samples = decoded_samples.view(B, T, decoded_samples.shape[-3], decoded_samples.shape[-2], decoded_samples.shape[-1])
            
            loss = criterion(decoded_samples, x)
            running_loss += loss.item()
                            
            metrics = metrics_calculator(x.to('cpu'), decoded_samples.to('cpu'))
            ssim_scores.append(metrics['SSIM'].mean())
            psnr_scores.append(metrics['PSNR'].mean())
            lpips_scores.append(metrics['LPIPS'].mean())
            fvd_scores.append(metrics['FVD'])

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    avg_fvd = sum(fvd_scores) / len(fvd_scores)
    avg_loss = running_loss / len(val_dataloader)
    
    logging.info(f"Validation Loss: {avg_loss:.4f}, "
          f"Validation SSIM: {avg_ssim:.4f}, "
          f"Validation PSNR: {avg_psnr:.4f}, "
          f"Validation LPIPS: {avg_lpips:.4f}, "
          f"Validation FVD: {avg_fvd:.4f} ")


def decode_in_batches(samples, vae, chunk_size=128):
    decoded_chunks = []
    num_chunks = (samples.shape[0] + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, samples.shape[0])
        chunk = samples[start_idx:end_idx]
        decoded_chunk = vae.decode(chunk).sample
        decoded_chunks.append(decoded_chunk)
        torch.cuda.empty_cache()

    return torch.cat(decoded_chunks, dim=0)

def main(args=None):
    ddp_setup()
    device = int(os.environ["LOCAL_RANK"])
    
    model = VDT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        mode='video'
    ).to(device)
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps), training=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model = DDP(model, device_ids=[device])

    train_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/train', transform=transforms.ToTensor())
    val_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/val', transform=transforms.ToTensor())

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, num_workers=4)

    train_vdt(model, train_dataloader, val_dataloader, vae, diffusion, optimizer, device, num_epochs=args.epoch, cfg_scale=1.0)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-L/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--f", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default="model.pt", help="Optional path to a VDT checkpoint.")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epoch', type=int, default=2)
    args = parser.parse_args()
    setup_logging()
    print("Number of process: ", torch.cuda.device_count())
    main(args=args)
