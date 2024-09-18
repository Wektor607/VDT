# All videos have 30 frames length in CityScapes
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch

class FrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=30):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.video_clips = []
        
        # Iterate through each subdirectory (city) and collect image paths
        for city in sorted(os.listdir(root_dir)):
            city_path = os.path.join(root_dir, city)
            if os.path.isdir(city_path):
                for sequence in sorted(os.listdir(city_path)):
                    sequence_path = os.path.join(city_path, sequence)
                    self.video_clips.append(sequence_path)
    
    def __len__(self):
        # Return the number of video clips
        return len(self.video_clips) // 30 #self.frames_per_clip

    def __getitem__(self, idx):
        # Determine the start and end index for the given video clip
        start_idx = idx * 30 #self.frames_per_clip #30
        end_idx = start_idx + 30 #self.frames_per_clip #30
        
        # Extract paths to 30 images for the given video clip
        clip_paths = self.video_clips[start_idx:end_idx]
        
        idx = np.random.randint(len(clip_paths) - self.frames_per_clip + 1)
        clip_paths = clip_paths[idx:idx + self.frames_per_clip]
        
        images = []

        for img_path in clip_paths:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        # Convert the list of images into a tensor of shape [30, C, H, W]
        images = torch.stack(images, dim=0)
        return images

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    train_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/train', transform=transform,frames_per_clip=16)

    # for idx, frames in enumerate(train_dataset):
    #     print(f"Video sequence {idx+1}:")
    #     print(f"Frame shape (C, H, W): {frames.shape}")
    #     print(f"Tensor shape: {frames.shape}")

    val_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/val', transform=transform,frames_per_clip=16)
    test_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/test', transform=transform,frames_per_clip=16)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)