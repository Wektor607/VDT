import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        # print(image.size)
        if self.transform:
            image = self.transform(image)
        return image, img_path

if __name__ == "__main__":
    original_folder = 'leftImg8bit_sequence' # Change the path here
    new_folder = original_folder + '_preprocess'
    os.makedirs(new_folder, exist_ok=True)

    transform = transforms.Compose([
        transforms.CenterCrop((2048, 1024)),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = ImageFolderDataset(root_dir=original_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for images, img_paths in dataloader:
        print(img_paths)
        for image, img_path in zip(images, img_paths):
            image = transforms.ToPILImage()(image)
            
            relative_path = os.path.relpath(img_path, original_folder)
            new_img_path = os.path.join(new_folder, relative_path)
            new_img_dir = os.path.dirname(new_img_path)
            
            os.makedirs(new_img_dir, exist_ok=True)
            
            image.save(new_img_path)
