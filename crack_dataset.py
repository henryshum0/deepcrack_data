from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import cv2
import data_proccess.data_generation as D

class CrackDataset(Dataset):
    def __init__(self, img_path:str=None, mask_path:str=None, transforms=[D.Resize((448,448))]):
        
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
            raise FileExistsError(f"No image files found in {self.img_path}.")
        if len(self.mask_files) == 0:
            raise FileExistsError(f"No mask files found in {self.mask_path}.")
        if len(self.img_files) != len(self.mask_files):
            raise FileExistsError(f"Number of images ({len(self.img_files)}) does not match number of masks ({len(self.mask_files)}).")
        
        self.data = list(zip(self.img_files, self.mask_files))
        self.data = sorted(self.data, key=lambda x: int(x[0].stem))
        self.transform_pipeline = D.DataGenPipeline(transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, mask_file = self.data[idx]
        image = cv2.imread(str(img_file), cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if self.transform_pipeline:
            image, mask = self.transform_pipeline(image, mask)
        return transforms.ToTensor()(image), transforms.ToTensor()(mask)
    
if __name__ == "__main__":
    
    #run the formatting.py first
    
    img_path = "test/images"
    mask_path = "test/masks"
    transforms_list = [
        
    ]
    dataset = CrackDataset(img_path=img_path, mask_path=mask_path, transforms=transforms_list)
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        img, mask = dataset[i]
        print(f"Image {i} shape: {img.shape}, Mask {i} shape: {mask.shape}")
        # Optionally, you can visualize the images and masks using matplotlib or any other library.
        # For example:
        cv2.imshow("Image", img.numpy().transpose(1, 2, 0))
        cv2.imshow("Mask", mask.numpy().transpose(1, 2, 0))
        cv2.waitKey(0)
        # Note: The above visualization code is commented out to avoid unnecessary imports.
        # Uncomment it if you want to visualize the images and masks.  