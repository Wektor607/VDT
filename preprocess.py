import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
# class ImageFolderDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []
#         for root, _, files in os.walk(root_dir):
#             for file in files:
#                 if file.endswith(('.png', '.jpg', '.jpeg')):
#                     self.image_paths.append(os.path.join(root, file))

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         # print(image.size)
#         if self.transform:
#             image = self.transform(image)
#         return image, img_path

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
        return len(self.video_clips) // self.frames_per_clip

    def __getitem__(self, idx):
        # Determine the start and end index for the given video clip
        start_idx = idx * self.frames_per_clip
        end_idx = start_idx + self.frames_per_clip
        
        # Extract paths to 30 images for the given video clip
        clip_paths = self.video_clips[start_idx:end_idx]
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

    train_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/train', transform=transform)

    # for idx, frames in enumerate(train_dataset):
    #     print(f"Video sequence {idx+1}:")
    #     print(f"Frame shape (C, H, W): {frames.shape}")
    #     print(f"Tensor shape: {frames.shape}")

    val_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/val', transform=transform)
    test_dataset = FrameDataset(root_dir='leftImg8bit_sequence_preprocess/test', transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # # Инициализация модели, оптимизатора и других компонентов
    # optimizer = Adam(model.parameters(), lr=1e-4)

    # # Запуск тренировочного процесса
    # train_vdt(model, train_dataloader, vae, diffusion, optimizer, device=torch.device('cuda'), num_epochs=20)
