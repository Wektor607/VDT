import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from models import VDT_models
from utils import load_checkpoint
from preprocess import FrameDataset
from diffusion import create_diffusion
from mask_generator import VideoMaskGenerator

def train_vdt(model, train_dataloader, val_dataloader, 
              vae, diffusion, optimizer, device, num_epochs=10, 
              cfg_scale=1.0):
    """
    Training loop for VDT model.

    Parameters:
    - model: The VDT model.
    - train_dataloader: DataLoader for training data.
    - val_dataloader: DataLoader for validation data.
    - vae: Pre-trained VAE model for encoding/decoding images.
    - diffusion: Diffusion model.
    - optimizer: Optimizer (e.g., Adam).
    - device: Device to run the training on (e.g., 'cuda' or 'cpu').
    - num_epochs: Number of epochs for training.
    - cfg_scale: Scale for classifier-free guidance.

    - choice_index: Type of task
    """
    model.to(device)
    vae.to(device)
    
    criterion = nn.MSELoss()  # Assuming we're using MSE loss for simplicity

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Set model to training mode
        for batch_idx, x in enumerate(train_dataloader):
            B, T, C, H, W = x.shape
            x = x.to(device)  # Move data to device

            # Encode frames to latent space
            with torch.no_grad():
                latent_x = vae.encode(x.view(-1, C, H, W)).latent_dist.sample().mul_(0.18215)
            latent_x = latent_x.view(B, T, -1, latent_x.shape[-2], latent_x.shape[-1])
            
            # Generate mask
            # Each iteration randomly choose task
            choice_idx = random.randint(0, 6)
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]))
            mask = generator(B, device, idx=choice_idx)  # Assuming task type is 'predict'
            
            # Generate noise tensor z
            z = torch.randn(B, T, 4, latent_x.shape[-2], latent_x.shape[-1], device=device)
            z = z.permute(0, 2, 1, 3, 4)
            
            # Forward pass through the diffusion model
            sample_fn = model.forward
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, progress=False, device=device,
                raw_x=latent_x, mask=mask
            )
            samples = samples.permute(1, 0, 2, 3, 4) * mask + latent_x.permute(2, 0, 1, 3, 4) * (1-mask)
            samples = samples.permute(1, 2, 0, 3, 4).reshape(-1, 4, latent_x.shape[-2], latent_x.shape[-1]) / 0.18215
            
            # Decode generated samples back to image space
            decoded_chunks = []
            chunk_size = 256
            num_chunks = (samples.shape[0] + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, samples.shape[0])
                chunk = samples[start_idx:end_idx]
                decoded_chunk = vae.decode(chunk).sample
                decoded_chunks.append(decoded_chunk)

            decoded_samples = torch.cat(decoded_chunks, dim=0)
            decoded_samples = decoded_samples.view(B, T, decoded_samples.shape[-3], decoded_samples.shape[-2], decoded_samples.shape[-1])
            
            # Compute loss between original and generated images
            loss = criterion(decoded_samples, x)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for logging
            running_loss += loss.item()

            if batch_idx % 100 == 99:  # Log every 100 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        # Validation phase
        validate_vdt(model, val_dataloader, vae, diffusion, device)

    torch.save(model.state_dict(), 'vdt_model.pth')
    print("Training finished.")

def validate_vdt(model, val_dataloader, vae, diffusion, device):
    """
    Validation loop for VDT model.
    
    Parameters:
    - model: The VDT model.
    - val_dataloader: DataLoader for validation data.
    - vae: Pre-trained VAE model for encoding/decoding images.
    - diffusion: Diffusion model.
    - device: Device to run the validation on (e.g., 'cuda' or 'cpu').
    """
    model.eval()
    running_loss = 0.0
    criterion = nn.MSELoss()  # Assuming we're using MSE loss for simplicity
    
    with torch.no_grad():
        for _, x in enumerate(val_dataloader):
            B, T, C, H, W = x.shape
            x = x.to(device)
            
            latent_x = vae.encode(x.view(-1, C, H, W)).latent_dist.sample().mul_(0.18215)
            latent_x = latent_x.view(B, T, -1, latent_x.shape[-2], latent_x.shape[-1])
            
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]))
            mask = generator(B, device, idx=0)
            
            z = torch.randn(B, T, 4, latent_x.shape[-2], latent_x.shape[-1], device=device)
            z = z.permute(0, 2, 1, 3, 4)
            
            sample_fn = model.forward
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, progress=False, device=device,
                raw_x=latent_x, mask=mask
            )
            samples = samples.permute(1, 0, 2, 3, 4) * mask + latent_x.permute(2, 0, 1, 3, 4) * (1-mask)
            samples = samples.permute(1, 2, 0, 3, 4).reshape(-1, 4, latent_x.shape[-2], latent_x.shape[-1]) / 0.18215
            
            decoded_chunks = []
            chunk_size = 256
            num_chunks = (samples.shape[0] + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, samples.shape[0])
                chunk = samples[start_idx:end_idx]
                decoded_chunk = vae.decode(chunk).sample
                decoded_chunks.append(decoded_chunk)

            decoded_samples = torch.cat(decoded_chunks, dim=0)
            decoded_samples = decoded_samples.view(B, T, decoded_samples.shape[-3], decoded_samples.shape[-2], decoded_samples.shape[-1])
            
            loss = criterion(decoded_samples, x)
            running_loss += loss.item()

    avg_loss = running_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-L/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--f", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=16) # Set higher for better results! (max 1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default="model.pt",
                        help="Optional path to a VDT checkpoint.")
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/train', transform=transform)
    val_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/val', transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    latent_size = args.image_size // 8
    additional_kwargs = {'num_frames': args.num_frames,
    'mode': 'video'} 

    model = VDT_models[args.model](
    input_size=latent_size,
    num_classes=args.num_classes,
    **additional_kwargs)
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_vdt(model, train_dataloader, val_dataloader, vae, diffusion, optimizer, device, num_epochs=20, cfg_scale=1.0)
