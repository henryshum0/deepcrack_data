import cv2
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

def check_imgmask(img_dir: str, mask_dir: str):
    """
    Check if images and masks in the specified directories match in terms of filenames and dimensions.
    
    Args:
        img_dir (str): Path to the directory containing images.
        mask_dir (str): Path to the directory containing masks.
        
    Raises:
        ValueError: If the number of images and masks do not match or if any image-mask pair has different dimensions.
    """
    img_path = Path(img_dir)
    mask_path = Path(mask_dir)

    if not img_path.exists():
        raise FileNotFoundError(f"Image directory {img_path} does not exist.")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask directory {mask_path} does not exist.")

    img_files = sorted([f for f in img_path.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    mask_files = sorted([f for f in mask_path.glob("*") if f.suffix.lower() in ['.png']])

    if (len(img_files) != len(mask_files)):
        print("number of images and masks do not match.")
        print(f"Number of images: {len(img_files)}, Number of masks: {len(mask_files)}")
    
    for img_file in img_files:
        img = cv2.imread(str(img_file))
        mask_file = (mask_path / (str(img_file.stem) +  '_GT' + '.png'))
        if not mask_file.exists():
            print(f"Mask file {mask_file} does not exist for image {img_file.name}.")
            continue
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"mask for {img_file.name} does not exists")
            continue
        
        if img is None or mask is None:
            print(f"image {img_file.name} is not in correct format")
            continue

        if img.shape[:2] != mask.shape[:2]:
            print(f"Image {img_file.name} and mask {mask_file.name} have different dimensions: "
                  f"Image shape: {img.shape}, Mask shape: {mask.shape}")
            continue
        mask = np.stack([mask>0] * 3, axis=-1)
        img[mask] = 255
        print("displaying image and mask for", img_file.name)
        plt.imshow(img)
        plt.show()
        
        
if __name__ == "__main__":
    img_dir = "Crack/images"
    mask_dir = "Crack/gt"
    check_imgmask(img_dir, mask_dir)
    cv2.destroyAllWindows()