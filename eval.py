import os
import torch
import random
import logging
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image

from mask_generator import VideoMaskGenerator
from utils import decode_in_batches

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
            if B < args.batch_size:
                continue
            raw_x = x.to(device)
            x = x.view(-1, C, H, W).to(device)
            
            with torch.no_grad():
                latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            latent_x = latent_x.view(-1, T, 4, latent_x.shape[-2], latent_x.shape[-1])
            
            choice_idx = random.choice([0, 3])
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]), num_frames=T)
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
            if B < args.batch_size:
                continue

            raw_x = x.to(device)
            x = x.view(-1, C, H, W).to(device)
            
            with torch.no_grad():
                latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            latent_x = latent_x.view(-1, T, 4, latent_x.shape[-2], latent_x.shape[-1])
            
            choice_idx = random.choice([0, 3])
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]), num_frames=T)
            mask = generator(B, device, idx=choice_idx)
            
            z = torch.randn(B, T, 4, input_size, input_size, device=device).permute(0, 2, 1, 3, 4)
            latent_x = latent_x.to(device)

            # Write about it in the paper
            sample_fn = model.module.forward_with_cfg if hasattr(model, 'module') else model.forward_with_cfg
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, progress=False, device=device,
                raw_x=latent_x, mask=mask
            )

            samples = samples.permute(1, 0, 2, 3, 4) * mask + latent_x.permute(2, 0, 1, 3, 4) * (1-mask)
            samples = samples.permute(1, 2, 0, 3, 4).reshape(-1, 4, input_size, input_size) / 0.18215
            
            decoded_samples = decode_in_batches(samples, vae)
            decoded_samples = decoded_samples.reshape(-1, T, decoded_samples.shape[-3], decoded_samples.shape[-2], decoded_samples.shape[-1])

            # running_loss += loss.item()
            # if batch_idx != 0 and batch_idx % (len(test_dataloader) // args.batch_size) == 0:
            #     logging.info(f"Test Batch [{batch_idx+1}/{len(test_dataloader)}], "
            #                     f"Mean Loss: {running_loss / (len(test_dataloader) // args.batch_size):.4f}")
            #     running_loss = 0.0
            print(decoded_samples.shape, raw_x.shape)           
            loss = criterion(decoded_samples.to('cpu'), raw_x.to('cpu'))
            full_loss += loss.item()
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