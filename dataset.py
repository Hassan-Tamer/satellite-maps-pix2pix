import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

class Maps(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

if __name__ == "__main__":
    dataset = Maps(data_dir='maps/train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for input_image, target_image in dataloader:
        input_image = input_image.squeeze()
        target_image = target_image.squeeze()

        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(target_image)
        plt.title('Target Image')
        plt.axis('off')

        plt.show()