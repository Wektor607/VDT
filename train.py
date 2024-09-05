import os
import torch
import random
import argparse
import logging
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributed as dist
import matplotlib.image as mpimg
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from models import VDT_models
from preprocess import FrameDataset
from diffusion import create_diffusion
from metrics import MetricCalculator
from diffusers.models import AutoencoderKL
from mask_generator import VideoMaskGenerator
from utils import load_checkpoint
from utils import setup_logging, ddp_setup, decode_in_batches
import torch.optim.lr_scheduler as lr_scheduler

def train_vdt(args, model, train_dataloader, val_dataloader, 
              vae, diffusion, optimizer, device, metrics_calculator, num_epochs=10,
              cfg_scale=1.0):
    
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        epoch_loss = 0.0
        for batch_idx, x in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()

            B, T, C, H, W = x.shape
            x = x.view(-1, C, H, W).to(device)

            with torch.no_grad():
                latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            latent_x = latent_x.view(-1, T, 4, latent_x.shape[-2], latent_x.shape[-1])
            
            choice_idx = random.choice([0, 3])
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]))
            mask = generator(B, device, idx=choice_idx)
            
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()
            latent_x = latent_x.to(device)

            sample_fn = model.forward
            results = diffusion.training_losses(sample_fn, latent_x, t, 
                                                  noise=None, mask=mask)
            
            loss = results["loss"].mean()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if batch_idx % 80 == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], "
                                f"Mean Loss: {running_loss / 80:.4f}")
                running_loss = 0.0
                
        logging.info(f"Full Epoch [{epoch+1}/{num_epochs}], "f"Final Loss: {epoch_loss / len(train_dataloader):.4f}")
        
        validate_vdt(args, model, val_dataloader, vae, diffusion, device, metrics_calculator)

        # scheduler.step(epoch_loss / len(train_dataloader))
        
        diffusion.training = True
        model_state = model.module.state_dict()
        # model_state = model.state_dict()
        
        torch.save(model_state, f'vdt_model_lf_epoch_{epoch+1}.pt')
        logging.info(f"Model saved after epoch {epoch+1}")

    torch.save(model.module.state_dict(), 'vdt_model_lfW.pt')
    # torch.save(model.state_dict(), 'vdt_model_3.pt')
    print("Training finished.")

def validate_vdt(args, model, val_dataloader, vae, diffusion, device, metrics_calculator):
    model.eval()
    running_loss = 0.0
    criterion = nn.MSELoss()

    ssim_scores  = []
    psnr_scores  = []
    lpips_scores = []
    fvd_scores   = []
    diffusion.training = False
    input_size=args.image_size // 8
    with torch.no_grad():
        for batch_idx, x in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            B, T, C, H, W = x.shape
            raw_x = x.to(device)
            x = x.view(-1, C, H, W).to(device)
            
            with torch.no_grad():
                latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            latent_x = latent_x.view(-1, T, 4, latent_x.shape[-2], latent_x.shape[-1])
            
            choice_idx = random.choice([0, 3])
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]))
            mask = generator(B, device, idx=choice_idx)
            
            z = torch.randn(B, T, 4, input_size, input_size, device=device).permute(0, 2, 1, 3, 4)
            latent_x = latent_x.to(device)

            sample_fn = model.forward
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, progress=False, device=device,
                raw_x=latent_x, mask=mask
            )

            samples = samples.permute(1, 0, 2, 3, 4) * mask + latent_x.permute(2, 0, 1, 3, 4) * (1-mask)
            samples = samples.permute(1, 2, 0, 3, 4).reshape(-1, 4, input_size, input_size) / 0.18215

            decoded_samples = decode_in_batches(samples, vae)
            decoded_samples = decoded_samples.reshape(-1, T, decoded_samples.shape[-3], decoded_samples.shape[-2], decoded_samples.shape[-1])
            
            loss = criterion(decoded_samples.to('cpu'), raw_x.to('cpu'))
            running_loss += loss.item()
                            
            metrics = metrics_calculator(decoded_samples.to('cpu'), raw_x.to('cpu'))
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

def test_vdt(args, model, test_dataloader, vae, diffusion, device, metrics_calculator):
    img_dir = "res_test"
    os.makedirs(img_dir, exist_ok=True)
    model.eval()
    running_loss = 0.0
    criterion = nn.MSELoss()

    ssim_scores  = []
    psnr_scores  = []
    lpips_scores = []
    fvd_scores   = []
    diffusion.training = False
    input_size = args.input_size // 8
    with torch.no_grad():
        for batch_idx, x in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            B, T, C, H, W = x.shape
            raw_x = x.to(device)
            x = x.view(-1, C, H, W).to(device)
            
            with torch.no_grad():
                latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            latent_x = latent_x.view(-1, T, 4, latent_x.shape[-2], latent_x.shape[-1])
            
            choice_idx = random.choice([0, 3])
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]))
            mask = generator(B, device, idx=choice_idx)
            
            z = torch.randn(B, T, 4, input_size, input_size, device=device).permute(0, 2, 1, 3, 4)
            latent_x = latent_x.to(device)

            sample_fn = model.forward
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, progress=False, device=device,
                raw_x=latent_x, mask=mask
            )

            samples = samples.permute(1, 0, 2, 3, 4) * mask + latent_x.permute(2, 0, 1, 3, 4) * (1-mask)
            samples = samples.permute(1, 2, 0, 3, 4).reshape(-1, 4, input_size, input_size) / 0.18215
            
            decoded_samples = decode_in_batches(samples, vae)
            decoded_samples = decoded_samples.reshape(-1, T, decoded_samples.shape[-3], decoded_samples.shape[-2], decoded_samples.shape[-1])

            loss = criterion(decoded_samples.to('cpu'), raw_x.to('cpu'))
            running_loss += loss.item()
                            
            metrics = metrics_calculator(decoded_samples.to('cpu'), raw_x.to('cpu'))
            ssim_scores.append(metrics['SSIM'].mean())
            psnr_scores.append(metrics['PSNR'].mean())
            lpips_scores.append(metrics['LPIPS'].mean())
            fvd_scores.append(metrics['FVD'])

            mask_resized = F.interpolate(mask.float(), size=(raw_x.shape[-2], raw_x.shape[-1]), mode='nearest')
            mask_resized = mask_resized.unsqueeze(0).repeat(3, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)

            raw_x = raw_x.reshape(-1, T, raw_x.shape[-3], raw_x.shape[-2], raw_x.shape[-1])
            raw_x_masked = raw_x * (1 - mask_resized)

            comparison_images = torch.cat([raw_x_masked, decoded_samples], dim=1)

            output_file = os.path.join(img_dir, f'output_{batch_idx}_{choice_idx}.png')

            save_image(comparison_images.reshape(-1, comparison_images.shape[-3], comparison_images.shape[-2], comparison_images.shape[-1]), \
                       output_file, nrow=T, normalize=True, value_range=(-1, 1))

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    avg_fvd = sum(fvd_scores) / len(fvd_scores)
    avg_loss = running_loss / len(test_dataloader)
    
    logging.info(f"Test Loss: {avg_loss:.4f}, "
                 f"Test SSIM: {avg_ssim:.4f}, "
                 f"Test PSNR: {avg_psnr:.4f}, "
                 f"Test LPIPS: {avg_lpips:.4f}, "
                 f"Test FVD: {avg_fvd:.4f} ")
    
def main_paral(args=None):
    ddp_setup()
    device = int(os.environ["LOCAL_RANK"])
    model = VDT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        mode='video'
    )
    # model, _ = load_checkpoint(model, args.ckpt)
    model = model.to(device)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps), training=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    metrics_calculator = MetricCalculator(['SSIM', 'PSNR', 'LPIPS', 'FVD'], device=device)

    model = DDP(model, device_ids=[device])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/train', transform=transform)
    val_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/val', transform=transform)
    test_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/test', transform=transform)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler)

    train_vdt(args, model, train_dataloader, val_dataloader, vae, diffusion, optimizer, device, metrics_calculator, num_epochs=args.epoch, cfg_scale=1.0)
    test_vdt(args, model, test_dataloader, vae, diffusion, device, metrics_calculator)
    
    dist.destroy_process_group()

def main_single(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VDT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        mode='video'
    )#.to(device)
    # model, _ = load_checkpoint(model, args.ckpt)
    model = model.to(device)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps), training=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    metrics_calculator = MetricCalculator(['SSIM', 'PSNR', 'LPIPS', 'FVD'], device=device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/train', transform=transform)
    val_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/val', transform=transform)
    test_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/test', transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_vdt(args, model, train_dataloader, val_dataloader, vae, diffusion, optimizer, device, metrics_calculator, num_epochs=args.epoch, cfg_scale=1.0)
    metrics_calculator = MetricCalculator(['SSIM', 'PSNR', 'LPIPS', 'FVD'], device=device)
    test_vdt(args, model, test_dataloader, vae, diffusion, device, metrics_calculator)
    

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
    parser.add_argument("--ckpt", type=str, default="vdt_model.pt", help="Optional path to a VDT checkpoint.")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epoch', type=int, default=2)
    args = parser.parse_args()
    setup_logging()
    # print("Number of process: ", torch.cuda.device_count())
    main_paral(args=args)
