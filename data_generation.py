import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from pathlib import Path

class Resize():
    """
    Resize the image and mask to a specified size.
    """
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image=None, mask=None):
        assert image.shape[2] == 3, f"Image must have 3 channels (RGB), but got {image.shape[2]} channels"
        assert mask.ndim == 2, f"Mask must be a 2D array {mask.ndim}D, but got {mask.shape}"
        if image is not None:
            image = cv2.resize(image, self.size)
        if mask is not None:
            mask = cv2.resize(mask, self.size)
        return image, mask

class RandomCrop():
    """
    Randomly crop the image and mask to a square size.
    """
    def __call__(self, image=None, mask=None, center_x=None, center_y=None) :
        assert image.shape[2] == 3, f"Image must have 3 channels (RGB), but got {image.shape[2]} channels"
        assert mask.ndim == 2, f"Mask must be a 2D array {mask.ndim}D, but got {mask.shape}"
        
        #randomly select a crop length
        crop_len = min(image.shape[0], image.shape[1], randint(4,10) * 128)
        
        #if center is given, crop around the center
        if center_x is None or center_y is None:
            left = randint(0, image.shape[1] - crop_len)
            top = randint(0, image.shape[0] - crop_len)
            
        #else randomly crop from left to right and top to bottom
        else:
            left = max(0, center_x - crop_len // 2)
            top = max(0, center_y - crop_len // 2)
        right = min(left + crop_len, image.shape[1])
        bottom = min(top + crop_len, image.shape[0])
        cropped_image = image[top:bottom, left:right]
        cropped_mask = mask[top:bottom, left:right]
        return cropped_image, cropped_mask

class RandomRotate():
    def __call__(self, image=None, mask=None):
        assert image.shape[2] == 3, f"Image must have 3 channels (RGB), but got {image.shape[2]} channels"
        assert mask.ndim == 2, f"Mask must be a 2D array {mask.ndim}D, but got {mask.shape}"
        angle = np.random.randint(0, 360)
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        rotated_mask = cv2.warpAffine(mask, M, (image.shape[1], image.shape[0]))
        return rotated_image, rotated_mask

class RandomFlip():
    def __call__(self, image=None, mask=None):
        assert image.shape[2] == 3, f"Image must have 3 channels (RGB), but got {image.shape[2]} channels"
        assert mask.ndim == 2, f"Mask must be a 2D array {mask.ndim}D, but got {mask.shape}"
        
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)  # Horizontal flip
            mask = cv2.flip(mask, 1)
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)  # Vertical flip
            mask = cv2.flip(mask, 0)
        
        return image, mask
        
class RandomCropResize():
    def __init__(self, has_mask:float = 0.9, size=(256, 256)):
        self.has_mask = has_mask
        self.resize = Resize(size)
        self.random_crop = RandomCrop()

    def __call__(self, image=None, mask=None):
        assert image.shape[2] == 3, f"Image must have 3 channels (RGB), but got {image.shape[2]} channels"
        assert mask.ndim == 2, f"Mask must be a 2D array {mask.ndim}D, but got {mask.shape}"
        
        if np.random.rand() < self.has_mask:
            mask_indices = np.argwhere(mask > 0)
            if mask_indices.size == 0:
                cropped_image, cropped_mask = self.random_crop(image, mask)
            else:
                center_y, center_x = mask_indices.mean(axis=0).astype(int)
                center_y += int(randint(-3, 3) * image.shape[0]/640 * 50)
                center_x += int(randint(-3, 3) * image.shape[1]/640 * 50)
                center_x = min(max(0, center_x), image.shape[1] - 256)
                center_y = min(max(0, center_y), image.shape[0] - 256)
                cropped_image, cropped_mask = self.random_crop(image, mask, center_x, center_y)
        else:
            cropped_image, cropped_mask = self.random_crop(image, mask)
        
        return self.resize(cropped_image, cropped_mask)
             
class RandomRotateRandomCropResize():
    def __init__(self, target_size=(256, 256)):
        self.resize = Resize(target_size)
        self.random_crop = RandomCrop()
        self.rotate = RandomRotate()
    def __call__(self, img, mask):
        rotated_img, rotated_mask = self.rotate(img, mask)
        cropped_img, cropped_mask = self.random_crop(rotated_img, rotated_mask)
        resized_img, resized_mask = self.resize(cropped_img, cropped_mask)
        return resized_img, resized_mask

class DataGenPipeline():
    def __init__(self, save:bool=False, load:bool=False, transforms=None, img_ld_dir:str=None, mask_ld_dir:str=None, 
                 img_save_dir:str=None, mask_save_dir:str=None):
        if load and (img_ld_dir is None  or mask_ld_dir is None):
            raise FileNotFoundError("Image and mask load directories must be specified.")
        if transforms is None or not isinstance(transforms, list) or transforms == []: 
            raise ValueError("Transforms must be a non-empty list of callable transformations.")
        
        if load:
            self.img_ld_dir = Path(img_ld_dir)
            self.mask_ld_dir = Path(mask_ld_dir)
            
        if save:
            if img_save_dir is None or mask_save_dir is None:
                img_save_dir = img_ld_dir + "/processed_imgs"
                mask_save_dir = mask_ld_dir + "/processed_masks"
            self.img_save_dir = Path(img_save_dir)
            self.mask_save_dir = Path(mask_save_dir)
            self.img_save_dir.mkdir(parents=True, exist_ok=True)
            self.mask_save_dir.mkdir(parents=True, exist_ok=True)
            
        self.load = load
        self.save = save
        self.transforms = transforms
        self.id = 0

    def __call__(self, image=None, mask=None):
        #case: when both load and save are true, pipeline processes all images and masks in the directories
        #and saves them to the save directories, then sets load and save to False
        if self.load and self.save and image is None and mask is None:
            print("pipeline is in loand and save")
            for img, msk in self.load_transfrom_data():
                self.save_data(img, msk)
            self.save = False
            self.load = False
            return True
        
        #case: when load is true, pipeline acts like a generator, yielding processed images and masks
        elif self.load:
            print("pipeline is in load mode")
            return self.load_transfrom_data()
        
        elif image is None or mask is None: 
            raise ValueError("Image and mask must be provided for processing.")
        
        #case: when save is true, pipeline applies transformations and saves the processed images and masks
        #to the save directories, then increments the id for the next save
        elif self.save :
            print("pipeline is in save mode")
            image, mask = self.apply_transforms(image, mask)
            self.save_data(image, mask)
            return True
        
        #case: only when image and mask are specified, pipeline applies transformations
        else:
            print("pipeline is in normal mode")
            return self.apply_transforms(image, mask)
    
    def apply_transforms(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask
    
    def save_data(self, image, mask):
        if self.save:
            img_path = self.img_save_dir / f"{self.id}.png"
            mask_path = self.mask_save_dir / f"{self.id}.png"
            cv2.imwrite(str(img_path), image)
            cv2.imwrite(str(mask_path), mask)
            print(f"Saved {img_path} and {mask_path}")
            print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
            self.id += 1
        else:
            raise RuntimeError("Save mode is not enabled. Cannot save data.")
        
    def load_transfrom_data(self):
        if self.load:
            img_files = []
            mask_files = []
            for ext in ["png", "jpg", "jpeg"]:
                img_files.extend(self.img_ld_dir.glob(f"*.{ext}"))
                mask_files.extend(self.mask_ld_dir.glob(f"*.{ext}"))
            img_files = sorted(img_files)
            mask_files = sorted(mask_files)
            if len(img_files) != len(mask_files):
                raise ValueError("Number of images and masks do not match.")
            for img_file, mask_file in zip(img_files, mask_files):
                image = cv2.imread(str(img_file))
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                print(f"Loading {img_file} and {mask_file}")
                print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
                yield self.apply_transforms(image, mask)
        else:
            raise RuntimeError("Load mode is not enabled. Cannot load data.")
        

    
    
    
if __name__ == "__main__":
    img = cv2.imread('test/imgs/0.png')
    mask = cv2.imread('test/masks/0.png', cv2.IMREAD_GRAYSCALE)
    
    img_ld_dir = 'test/imgs'
    mask_ld_dir = 'test/masks'
    
    img_save_dir = 'test/processed_imgs'
    mask_save_dir = 'test/processed_masks'
    
    transforms = [RandomCrop(), Resize((256, 256))]
    
    # #testing save only
    # pipeline = DataGenPipeline(save=True, load=False, transforms=transforms,
    #                           img_ld_dir=img_ld_dir, mask_ld_dir=mask_ld_dir,
    #                           img_save_dir=img_save_dir, mask_save_dir=mask_save_dir)
    # pipeline(img, mask)
    
    # #testing load only
    # pipeline = DataGenPipeline(save=False, load=True, transforms=transforms,
    #                           img_ld_dir=img_ld_dir, mask_ld_dir=mask_ld_dir)

    # for img, mask in pipeline():
    #     print("hi")
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img)
    #     print(img.shape)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(mask, cmap='gray')
    #     print(mask.shape)
    #     plt.show(block=True)
        
    # #testing both load and save
    # pipeline = DataGenPipeline(save=True, load=True, transforms=transforms,
    #                           img_ld_dir=img_ld_dir, mask_ld_dir=mask_ld_dir,
    #                           img_save_dir=img_save_dir, mask_save_dir=mask_save_dir)
    # print(pipeline())
    
    # #testing on the fly transformation
    # pipeline = DataGenPipeline(save=False, load=False, transforms=transforms) 
    # img_aug, mask_aug = pipeline(img, mask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_aug)
    # print(img_aug.shape)
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask_aug, cmap='gray')
    # print(mask_aug.shape)
    # plt.show(block=True)