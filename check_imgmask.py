from PIL import Image
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import cv2

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
    out_imgs = []
    original_imgs = []
    out_img_names = []

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
        img = np.array(Image.open(str(img_file))).astype(np.uint8)
        mask_file = (mask_path / (str(img_file.stem) +  '_GT' + '.png'))
        if not mask_file.exists():
            print(f"Mask file {mask_file} does not exist for image {img_file.name}.")
            continue
        mask = np.array(Image.open(str(mask_file)).convert('L')).astype(np.uint8)
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
        bool_idx = mask > 0
        out_img = img.copy()
        out_img[bool_idx] = [0, 255, 0]
        print(f"Image {img_file.name} and mask {mask_file.name} are matched.")
        out_imgs.append(cv2.resize(out_img, (1280, 720)))
        out_img_names.append(img_file.name)
        original_imgs.append(cv2.resize(img, (1280, 720)))
    idx = 0
    display_original = False
    while 0 <= idx < len(out_imgs):
        if display_original:
            cv2.imshow("image", original_imgs[idx])
        else:
            cv2.imshow("image", out_imgs[idx])
        print(f"Image: {out_img_names[idx]}")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'):  # Next image
            idx += 1
            idx = min(idx, len(out_imgs) - 1)
        elif key == ord('a'):  # Previous image
            idx -= 1
            idx = max(0, idx)
        elif key == ord('s'):
            display_original = not display_original
        elif key == 27:  # ESC to exit
            break
    cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    img_dir = "Crack/images"
    mask_dir = "Crack/gt"
    check_imgmask(img_dir, mask_dir)