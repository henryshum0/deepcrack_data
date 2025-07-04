import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from pathlib import Path
import os

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

class RandomBoostContrast():
    """
    Boost the contrast of the image.
    """
    def __call__(self, image=None, mask=None):
        assert image is not None, "Image must be provided for contrast boosting."
        assert image.shape[2] == 3, f"Image must have 3 channels (RGB), but got {image.shape[2]} channels"
        alpha = np.random.uniform(1.3, 2)  # Contrast factor, can be adjusted
        # Convert to float32 for better precision
        image = image.astype(np.float32)
        
        # Boost the contrast by multiplying by a factor
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)  # Adjust alpha for contrast
        
        return image, mask
    
class RandomJitter():
    """
    Randomly adjust brightness, contrast, saturation, and hue of the image.
    """
    def __call__(self, image=None, mask=None):
        assert image is not None, "Image must be provided for random jitter."
        assert image.shape[2] == 3, f"Image must have 3 channels (RGB), but got {image.shape[2]} channels"
        
        # Randomly adjust brightness
        brightness = np.random.uniform(0.5, 1.5)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Randomly adjust contrast
        contrast = np.random.uniform(0.5, 1.5)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        
        # Randomly adjust saturation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * np.random.uniform(0.5, 1.5), 0, 255)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        # Randomly adjust hue
        hsv_image[:, :, 0] = ((hsv_image[:, :, 0].astype(int) + np.random.randint(-10, 10)) % 180).astype(np.uint8)  # Hue values are in [0, 180] for OpenCV
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        return image, mask
    
class RandomGaussianNoise():
    """
    Add random Gaussian noise to the image.
    """
    def __call__(self, image=None, mask=None):
        assert image is not None, "Image must be provided for adding Gaussian noise."
        assert image.shape[2] == 3, f"Image must have 3 channels (RGB), but got {image.shape[2]} channels"
        
        sigma = np.random.uniform(1, 6)  # Randomly select a sigma value for noise
        noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        
        return noisy_image, mask

class RandomGaussianBlur():
    """
    Apply random Gaussian blur to the image.
    """
    
    def __call__(self, image=None, mask=None, ksize=(3, 3)):
        assert image is not None, "Image must be provided for Gaussian blur."
        assert image.shape[2] == 3, f"Image must have 3 channels (RGB), but got {image.shape[2]} channels"
        
        blurred_image = cv2.GaussianBlur(image, ksize, sigmaX = 3, sigmaY= 3)
        
        return blurred_image, mask

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
                center_y += int(randint(-3, 3) * image.shape[0]/640 * 25)
                center_x += int(randint(-3, 3) * image.shape[1]/640 * 25)
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
    
class RandomSequential():
    """
    Apply a random sequence of transformations to the image and mask.
    """
    def __init__(self, transforms, p=0.3):
        self.transforms = transforms
        self.p = 0.3

    def __call__(self, image=None, mask=None):
        assert image is not None, "Image must be provided for random sequence of transformations."
        assert mask is not None, "Mask must be provided for random sequence of transformations."
        
        for transform in self.transforms:
            if np.random.rand() < self.p:  # Randomly apply each transformation
                assert callable(transform), f"Transform {transform} is not callable."
                if isinstance(transform, (Resize, RandomCrop, RandomRotate, RandomFlip, RandomBoostContrast, 
                                          RandomJitter, RandomGaussianNoise, RandomGaussianBlur)):
                    image, mask = transform(image, mask)
                else:
                    raise ValueError(f"Unsupported transform type: {type(transform)}")
        return image, mask

class DataGenPipeline():
    def __init__(self, save:bool=False, load:bool=False, transforms=[], img_ld_dir:str=None, mask_ld_dir:str=None, 
                 img_save_dir:str=None, mask_save_dir:str=None, count:int=1, 
                 mask_suffix:str='', save_prefix:str='', save_suffix:str='', save_mask_suffix:str='', start_idx:int=0):
        '''
        if both save and load are true, pipeline processes all images and masks in the directories
        and saves them to the save directories, then sets load and save to False. Then it acts
        like a generator, yielding processed images and masks.
        if only load is true, pipeline acts like a generator, yielding processed images and masks.
        if only save is true, pipeline applies transformations and saves the processed images and masks
        to the save directories, then increments the id for the next save.
        
        Args:
            save (bool): Whether to save the processed images and masks.
            load (bool): Whether to load and process images and masks from the directories.
            transforms (list): List of transformations to apply to the images and masks.
            img_ld_dir (str): Directory to load images from.
            mask_ld_dir (str): Directory to load masks from.
            img_save_dir (str): Directory to save processed images to.
            mask_save_dir (str): Directory to save processed masks to.
            count (int):Number of times of looping through images to generate data.
                        only applicable when load is true.
            mask_suffix (str):  Suffix to append to the mask filenames when loading. 
                                example: img: "MyData_1.png", mask: "MyData_1_GT.png", herer "_GT" is the suffix.
            save_prefix (str): Prefix to prepend to the saved image and mask filenames.
            save_suffix (str): Suffix to append to the saved image and mask filenames.
            save_mask_suffix (str): Suffix to append to the saved mask filenames.
                                    example: img: "save_1_aug.png", mask: "save_1_aug_GT.png", here "_GT" is the save_mask_suffix. 
        '''
        
        if load:
            self.img_ld_dir = Path(img_ld_dir)
            self.mask_ld_dir = Path(mask_ld_dir)
            if not self.img_ld_dir.exists():
                raise FileNotFoundError(f"Image load directory {self.img_ld_dir} does not exist.")
            if not self.mask_ld_dir.exists():
                raise FileNotFoundError(f"Mask load directory {self.mask_ld_dir} does not exist.")
        # if transforms is None or not isinstance(transforms, list) or transforms == []: 
        #     raise ValueError("Transforms must be a non-empty list of callable transformations.")
            
            
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
        self.id = start_idx
        self.count = count
        self.mask_suffix = mask_suffix
        self.save_prefix = save_prefix
        self.save_suffix = save_suffix
        self.save_mask_suffix = save_mask_suffix

    def __call__(self, image=None, mask=None):
        #case: when both load and save are true, pipeline processes all images and masks in the directories
        #and saves them to the save directories, then sets load and save to False
        if self.load and self.save and image is None and mask is None:
            print("pipeline is in loand and save")
            for transformed, _ in self.load_transfrom_data():
                img, msk = transformed
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
            img_path = self.img_save_dir / f"{self.save_prefix}{self.id}{self.save_suffix}.png"
            mask_path = self.mask_save_dir / f"{self.save_prefix}{self.id}{self.save_suffix}{self.save_mask_suffix}.png"
            cv2.imwrite(str(img_path), image)
            mask = (mask / 255).astype(np.uint8)*255  # Ensure mask is in the correct format
            assert np.unique(mask).all() in [0, 1, 255], "Mask should be binary or grayscale with values 0, 1, or 255."
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
            for ext in ["png", "jpg", "jpeg", "JPG", "JPEG", "PNG"]:
                img_files.extend(self.img_ld_dir.glob(f"*.{ext}"))
                mask_files.extend(self.mask_ld_dir.glob(f"*.{ext}"))
            if len(img_files) != len(mask_files):
                raise ValueError("Number of input images and masks do not match." f"{len(img_files)} images and {len(mask_files)} masks found.")
            while self.count > 0:
                for img_file in img_files:
                    image = cv2.imread(str(img_file))
                    mask_file = self.mask_ld_dir / (img_file.stem + self.mask_suffix + ".png")
                    # print(f"Loading {img_file} and {mask_file}")
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    mask = (mask / 255).round().astype(np.uint8) * 255  # Ensure mask is in the correct format
                    # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
                    yield self.apply_transforms(image, mask), (img_file, mask_file)
                self.count -= 1
        else:
            raise RuntimeError("Load mode is not enabled. Cannot load data.")
        


if __name__ == "__main__":
    img = cv2.imread('test/imgs/0.png')
    mask = cv2.imread('test/masks/0.png', cv2.IMREAD_GRAYSCALE)
    
    crack_img_dir = "Crack/images"
    crack_mask_dir = "Crack/gt"
    crack_imgts_dir = "Crack/test_img"
    crack_maskts_dir = "Crack/test_label"
    
    img_ld_dir = 'test/imgs'
    mask_ld_dir = 'test/masks'
    
    img_save_dir = 'test/processed_imgs'
    mask_save_dir = 'test/processed_masks'
    
    aug_img_dir = 'test/contrast_imgs'
    aug_mask_dir = 'test/contrast_masks'
    
    # noncontrast_img_dir = 'test/noncontrast_imgs'
    # noncontrast_mask_dir = 'test/noncontrast_masks'
    
    transforms = [  #transfroms to apply
        RandomSequential([        #randomly select and apply transfroms
            RandomRotate(),
            RandomFlip(),
            RandomJitter(),
            RandomGaussianNoise(),
            RandomGaussianBlur(),
        ], p=0.8),
        RandomCropResize(size=(448, 448)),  #resize after cropping
    ]                  
    
    # #testing save only
    # pipeline = DataGenPipeline(save=True, load=False, transforms=transforms,
    #                           img_ld_dir=img_ld_dir, mask_ld_dir=mask_ld_dir,
    #                           img_save_dir=img_save_dir, mask_save_dir=mask_save_dir)
    # pipeline(img, mask)
    
    #testing load only
    # pipeline = DataGenPipeline(save=False, load=True, transforms=transforms,
    #                           img_ld_dir=img_ld_dir, mask_ld_dir=mask_ld_dir)

    # for transfromed, files in pipeline():
    #     img, mask = transfromed
    #     img_file, mask_file = files
    #     og_img = cv2.resize(cv2.imread(img_file), (448, 448))
    #     og_mask = cv2.resize(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE), (448, 448))
    #     print(f"Loaded {img_file} and {mask_file}")
    #     print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
        
        # img_out = np.concatenate([og_img, img], axis=1)  # Concatenate original and transformed images
        # mask_out = np.concatenate([og_mask, mask], axis=1)
        # mask_out = np.stack([mask_out] * 3, axis=-1)  # Convert mask to 3D for concatenation
        # out_img = np.concatenate([img_out, mask_out], axis=0)  # Concatenate images and masks vertically
        # cv2.imshow("Transformed Image and Mask", out_img)
        # cv2.waitKey(500)
        
    #testing both load and save
    howard_dir_img = "howard/images"
    howard_dir_mask = "howard/gt"
    howard_ts_img = "howard/images_ts"
    howard_ts_mask = "howard/gt_ts"
    
    transforms = [
        RandomSequential([        #randomly select and apply transfroms
            RandomRotate(),
            RandomFlip(),
            RandomJitter(),
            RandomGaussianNoise(),
            RandomGaussianBlur(),
        ], p=0.8),
        Resize(size=(448, 448)),  #resize after cropping
    ]
    
    pipeline = DataGenPipeline(save=True, load=True, transforms=transforms,
                              img_ld_dir=crack_imgts_dir, mask_ld_dir=crack_maskts_dir,
                              img_save_dir=howard_ts_img, mask_save_dir=howard_ts_mask, count=5, 
                              mask_suffix="_GT", save_prefix="", save_suffix="", save_mask_suffix="_GT", start_idx = 120)
    print(pipeline())
    
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
    
    # ld_dir = Path(img_ld_dir)
    # mk_dir = Path(mask_ld_dir)
    # img_files = ld_dir.glob('*.png')
    # pipeline1 = DataGenPipeline(save=False, load=False, transforms=transforms,
    #                             img_ld_dir=img_ld_dir, mask_ld_dir=mask_ld_dir,
    #                             img_save_dir=img_save_dir, mask_save_dir=mask_save_dir, count=1)
    # pipeline2 = DataGenPipeline(save=False, load=False, transforms=[Resize((448,448))],
    #                             img_ld_dir=img_ld_dir, mask_ld_dir=mask_ld_dir,
    #                             img_save_dir=img_save_dir, mask_save_dir=mask_save_dir, count=1)
    # #pipelines are created to transfrom images
    # # when pipeline is input with image and mask, the transfroms are applied
    # # on the fly. 
    # out_img = None
    # plt.figure(figsize=(20, 20))
    # for i in range(4):
    #     img_file = str(next(img_files))
    #     mask_file = str(mk_dir / img_file.split('/')[-1])
    #     print(img_file, mask_file)
    #     img = cv2.imread(img_file)
    #     mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
    #     img_resized, mask_resized = pipeline2(img, mask)
    #     img_aug, mask_aug = pipeline1(img_resized, mask_resized)
        
    #     mask_3d = np.stack([mask_resized] * 3, axis=-1)  # Convert mask to 3D for concatenation
    #     mask_aug_3d = np.stack([mask_aug] * 3, axis=-1)  # Convert mask to 3D for concatenation
    #     if out_img is None:
    #         out_img = np.concatenate([img_resized, mask_3d, img_aug, mask_aug_3d], axis=1)
    #     else:
    #         iiiimg = np.concatenate([img_resized, mask_3d, img_aug, mask_aug_3d], axis=1)
    #         out_img = np.concatenate([out_img, iiiimg], axis=0)
            
    #     plt.imshow(out_img)
    #     plt.axis('off')
    
    # plt.show(block=True)


# On the left are original resized images
# On the right are augmented images with masks
        