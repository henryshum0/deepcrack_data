import cv2, os
import numpy as np
from random import randint, uniform

def random_rotate_transform(image, angle_range=(-30, 30)):
    angle = uniform(*angle_range)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return M

def random_crop_transform(image, crop_size=(128, 128)):
    h, w = image.shape[:2]
    ch, cw = crop_size
    if h < ch or w < cw:
        raise ValueError("Crop size must be smaller than image size.")
    x = randint(0, w - cw)
    y = randint(0, h - ch)
    return (x, y, cw, ch)

def random_jitter(image, jitter=30):
    # Add random brightness jitter
    jitter_val = randint(-jitter, jitter)
    img = np.clip(image.astype(np.int16) + jitter_val, 0, 255).astype(np.uint8)
    return img, jitter_val

def random_gaussian_noise(image, mean=0, std=10):
    # Add Gaussian noise
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_img = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img, noise

def random_contrast(image, lower=0.5, upper=1.5):
    # Apply random contrast adjustment
    factor = uniform(lower, upper)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    img = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    return img, factor

img_dir = "Crack/images"
extents = [".jpg", ".jpeg", ".png"]
for ext in extents:
    img_files = [f for f in os.listdir(img_dir) if f.endswith(ext)]
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)
        print(f"Processing {img_file}...")
        
        # Apply random rotation
        M = random_rotate_transform(image)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Apply random crop
        x, y, cw, ch = random_crop_transform(rotated_image, (512,512))
        cropped_image = rotated_image[y:y+ch, x:x+cw]
        
        # Apply random jitter
        jittered_image, jitter_val = random_jitter(cropped_image)
        
        # Apply random Gaussian noise
        noisy_image, noise = random_gaussian_noise(jittered_image)
        
        # Apply random contrast
        contrast_image, contrast_factor = random_contrast(noisy_image)
        
        # Draw crop rectangle (after rotation) on original image
        # 1. Crop corners in rotated image
        crop_corners = np.array([
            [x, y],
            [x+cw, y],
            [x+cw, y+ch],
            [x, y+ch]
        ], dtype=np.float32)
        # 2. Inverse rotation matrix
        M_inv = cv2.invertAffineTransform(M)
        # 3. Map corners back to original image
        crop_corners_orig = cv2.transform(crop_corners[None, :, :], M_inv)[0]
        # 4. Draw polygon on original image
        original_image = cv2.resize(image, (512, 512))
        draw_img = image.copy()
        pts = np.int32(crop_corners_orig)
        cv2.polylines(draw_img, [pts], isClosed=True, color=(0,255,0), thickness=30)
        draw_img = cv2.resize(draw_img, (512, 512))

        contrast_image = cv2.resize(contrast_image, (512,512))
        cv2.imshow("Original Image with Crop+Rotation", draw_img)
        cv2.imshow("Transformed Image", contrast_image)
        cv2.waitKey(1000)