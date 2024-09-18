import os
import torch
import random
import argparse
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
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
from utils import setup_logging, ddp_setup, decode_in_batches, load_checkpoint, add_border
import torch.optim.lr_scheduler as lr_scheduler

def append_to_npy(file_path, new_data):
    if os.path.exists(file_path):
        existing_data = np.load(file_path)
        updated_data = np.concatenate((existing_data, new_data))
    else:
        updated_data = new_data

    np.save(file_path, updated_data)

def train_vdt(args, model, train_dataloader, val_dataloader, 
              vae, diffusion, optimizer, device, metrics_calculator, num_epochs=10,
              cfg_scale=1.0):
    final_epoch = 345
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_losses = []
    val_losses = []
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

            epoch_loss += loss.item()
            
            # running_loss += loss.item()
            # if batch_idx != 0 and batch_idx % (len(train_dataloader) // args.batch_size) == 0:
            #     logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], "
            #                     f"Mean Loss: {running_loss / (len(train_dataloader) // args.batch_size):.4f}")
            #     running_loss = 0.0
        train_losses.append(epoch_loss / len(train_dataloader))    
        logging.info(f"Full Epoch [{final_epoch + epoch+1}/{final_epoch + num_epochs}], "f"Final Loss: {epoch_loss / len(train_dataloader):.4f}")
        
        if epoch % 5 == 0 and epoch != 0:
            torch.save(model_state, f'vdt_model_{args.num_sampling_steps}_{final_epoch + epoch+1}.pt')
            logging.info(f"Model saved after epoch {final_epoch + epoch+1}")

            val_loss = validate_vdt(args, model, val_dataloader, vae, diffusion, device, metrics_calculator)
            val_losses.append(val_loss)

            # Append validation losses to the existing .npy file
            append_to_npy(f'val_losses_{args.num_sampling_steps}.npy', np.array(val_losses))

            scheduler.step(val_loss)

        # Append training losses to the existing .npy file
        append_to_npy(f'train_losses_{args.num_sampling_steps}.npy', np.array(train_losses))

        
        diffusion.training = True
        if args.mode == 'paral':
            model_state = model.module.state_dict()
        elif args.mode == 'single':
            model_state = model.state_dict()

    if args.mode == 'paral':
        torch.save(model.module.state_dict(), f'vdt_model_big_{args.num_sampling_steps}.pt')
    elif args.mode == 'single':
        torch.save(model.state_dict(), 'vdt_model_3.pt')
    print("Training finished.")

def validate_vdt(args, model, val_dataloader, vae, diffusion, device, metrics_calculator):
    model.eval()
    running_loss, full_loss = 0.0, 0.0
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
            full_loss += loss.item()
            # running_loss += loss.item()
            # if batch_idx != 0 and batch_idx % (len(val_dataloader) // args.batch_size) == 0:
            #     logging.info(f"Validation Batch [{batch_idx+1}/{len(val_dataloader)}], "
            #                     f"Mean Loss: {running_loss / (len(val_dataloader) // args.batch_size):.4f}")
            #     running_loss = 0.0
                            
            metrics = metrics_calculator(decoded_samples.to('cpu'), raw_x.to('cpu'))
            ssim_scores.append(metrics['SSIM'].mean())
            psnr_scores.append(metrics['PSNR'].mean())
            lpips_scores.append(metrics['LPIPS'].mean())
            fvd_scores.append(metrics['FVD'])

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    avg_fvd = sum(fvd_scores) / len(fvd_scores)
    avg_loss = full_loss / len(val_dataloader)
    
    append_to_npy(f'val_ssim_{args.num_sampling_steps}.npy', np.array(ssim_scores))
    append_to_npy(f'val_psnr_{args.num_sampling_steps}.npy', np.array(psnr_scores))
    append_to_npy(f'val_lpips_{args.num_sampling_steps}.npy', np.array(lpips_scores))
    append_to_npy(f'val_fvd_{args.num_sampling_steps}.npy', np.array(fvd_scores))
    
    logging.info(f"Validation Loss: {avg_loss:.4f}, "
          f"Validation SSIM: {avg_ssim:.4f}, "
          f"Validation PSNR: {avg_psnr:.4f}, "
          f"Validation LPIPS: {avg_lpips:.4f}, "
          f"Validation FVD: {avg_fvd:.4f} ")
    
    return avg_loss

def test_vdt(args, model, test_dataloader, vae, diffusion, device, metrics_calculator):
    img_dir = f"res_test_{args.ckpt}_8_8"
    os.makedirs(img_dir, exist_ok=True)
    model.eval()
    running_loss, full_loss = 0.0, 0.0
    criterion = nn.MSELoss()

    ssim_scores  = []
    psnr_scores  = []
    lpips_scores = []
    fvd_scores   = []
    diffusion.training = False
    input_size = args.image_size // 8
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

            # Write about it in the paper
            sample_fn = model.forward_with_cfg
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, progress=False, device=device,
                raw_x=latent_x, mask=mask
            )

            samples = samples.permute(1, 0, 2, 3, 4) * mask + latent_x.permute(2, 0, 1, 3, 4) * (1-mask)
            samples = samples.permute(1, 2, 0, 3, 4).reshape(-1, 4, input_size, input_size) / 0.18215
            
            decoded_samples = decode_in_batches(samples, vae)
            decoded_samples = decoded_samples.reshape(-1, T, decoded_samples.shape[-3], decoded_samples.shape[-2], decoded_samples.shape[-1])

            loss = criterion(decoded_samples.to('cpu'), raw_x.to('cpu'))
            full_loss += loss.item()
            # running_loss += loss.item()
            # if batch_idx != 0 and batch_idx % (len(test_dataloader) // args.batch_size) == 0:
            #     logging.info(f"Test Batch [{batch_idx+1}/{len(test_dataloader)}], "
            #                     f"Mean Loss: {running_loss / (len(test_dataloader) // args.batch_size):.4f}")
            #     running_loss = 0.0
            print(decoded_samples.shape, raw_x.shape)           
            metrics = metrics_calculator(decoded_samples.to('cpu'), raw_x.to('cpu'))
            ssim_scores.append(metrics['SSIM'].mean())
            psnr_scores.append(metrics['PSNR'].mean())
            lpips_scores.append(metrics['LPIPS'].mean())
            fvd_scores.append(metrics['FVD'])

            logging.info(f"Test Loss: {loss}, "
                f"Test SSIM: {metrics['SSIM'].mean()}, "
                f"Test PSNR: {metrics['PSNR'].mean()}, "
                f"Test LPIPS: {metrics['LPIPS'].mean()}, "
                f"Test FVD: {metrics['FVD']} ")
            # if choice_idx == 0:
            #     decoded_samples[0] = add_border(decoded_samples[0], color='orange')
            #     decoded_samples[1] = add_border(decoded_samples[1], color='orange')

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
    avg_loss = full_loss / len(test_dataloader)
    
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
        mode='video',
        cfg_scale=args.cfg_scale,
    )

    model, _ = load_checkpoint(model, args.ckpt)
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps), training=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    metrics_calculator = MetricCalculator(['SSIM', 'PSNR', 'LPIPS', 'FVD'], device=device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/train', transform=transform, frames_per_clip=args.num_frames)
    val_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/val', transform=transform, frames_per_clip=args.num_frames)
    test_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/test', transform=transform, frames_per_clip=args.num_frames)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler)

    if args.run_mode == 'train':
        train_vdt(args, model, train_dataloader, val_dataloader, vae, diffusion, optimizer, device, metrics_calculator, num_epochs=args.epoch, cfg_scale=1.0)#, writer=writer)
    
    test_vdt(args, model, test_dataloader, vae, diffusion, device, metrics_calculator)
    
    dist.destroy_process_group()

def main_single(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VDT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        mode='video',
        cfg_scale=args.cfg_scale,
    )
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
    parser.add_argument('--mode', type=str, default='single')
    parser.add_argument('--run_mode', type=str, default='train')
    args = parser.parse_args()
    
    setup_logging()
    
    if args.mode == 'paral':
        main_paral(args=args)
    elif args.mode == 'single':
        main_single(args=args)
    else:
        raise('There is no such mode!')