import os
import torch
import random
import logging
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

from eval import validate_vdt
from mask_generator import VideoMaskGenerator

def train_vdt(args, model, train_dataloader, val_dataloader, 
              vae, diffusion, optimizer, device, metrics_calculator, num_epochs=10):
    
    final_epoch = int(args.ckpt.rsplit('_', 1)[-1].split('.')[0])
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
            if B < args.batch_size:
                continue
            x = x.view(-1, C, H, W).to(device)

            with torch.no_grad():
                latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            latent_x = latent_x.view(-1, T, 4, latent_x.shape[-2], latent_x.shape[-1])
            
            choice_idx = random.choice([0, 3])
            generator = VideoMaskGenerator((latent_x.shape[-4], latent_x.shape[-2], latent_x.shape[-1]), num_frames=T)
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
            scheduler.step(val_loss)
        
        diffusion.training = True
        if args.mode == 'paral':
            model_state = model.module.state_dict()
        elif args.mode == 'single':
            model_state = model.state_dict()

    if args.mode == 'paral':
        torch.save(model.module.state_dict(), f'vdt_model_big_{args.num_sampling_steps}.pt')
    elif args.mode == 'single':
        torch.save(model.state_dict(), f'vdt_model_big_{args.num_sampling_steps}.pt')
    print("Training finished.")