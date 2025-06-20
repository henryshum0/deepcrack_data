from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import own_transfroms as T
from pathlib import Path
from PIL import Image

class CrackDataset(Dataset):
    def __init__(self, img_path:str=None, mask_path:str=None, transforms=[]):
        
        # checking if image and mask paths are valid
        self.img_path = Path(img_path)
        self.mask_path = Path(mask_path)
        if not self.img_path.exists():
            raise FileNotFoundError(f"Image path {self.img_path} does not exist.")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask path {self.mask_path} does not exist.")
        
        # getting all image and mask files
        exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        self.img_files = sorted([f for f in self.img_path.rglob("*") if f.suffix in exts and f.stem.isdigit()])
        self.mask_files = sorted([f for f in self.mask_path.rglob("*") if f.suffix in exts and f.stem.isdigit()])
        if len(self.img_files) == 0:
            raise ValueError(f"No image files found in {self.img_path}.")
        if len(self.mask_files) == 0:
            raise ValueError(f"No mask files found in {self.mask_path}.")
        if len(self.img_files) != len(self.mask_files):
            raise ValueError(f"Number of images ({len(self.img_files)}) does not match number of masks ({len(self.mask_files)}).")
        
        self.data = list(zip(self.img_files, self.mask_files))
        self.data = sorted(self.data, key=lambda x: int(x[0].stem))
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, mask_file = self.data[idx]
        image = transforms.ToTensor()(Image.open(img_file).convert("RGB"))
        mask = transforms.ToTensor()(Image.open(mask_file).convert("L"))

        if self.transforms:
            for transform in self.transforms:
                image, mask = transform(image, mask)


        return image, mask
    
if __name__ == "__main__":
    
    #run the formatting.py first
    
    img_path = "test/processed_imgs"
    mask_path = "test/processed_masks"
    transforms_list = [
        # T.random_crop,
        # T.random_horizontal_flip,
        # T.normalize,
        # T.random_affine,
        # T.random_rotation,
        # T.random_color_jitter,
        T.random_gaussian_blur
    ]
    dataset = CrackDataset(img_path=img_path, mask_path=mask_path, transforms=transforms_list)
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        img, mask = dataset[i]
        print(f"Image {i} shape: {img.shape}, Mask {i} shape: {mask.shape}")
        # Optionally, you can visualize the images and masks using matplotlib or any other library.
        # For example:
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.show()
        # Note: The above visualization code is commented out to avoid unnecessary imports.
        # Uncomment it if you want to visualize the images and masks.   