import os
import shutil
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt

def formatting_data(img_in_dir:str=None, label_in_dir:str=None, img_out_dir:str=None, label_out_dir:str=None, prefix:str=''):
    """
    Formatting the input images and labels to a standardized format.
    ensure that image is RGB png having 3 channels img.shape = (M, N, 3) 
    and label is binary mask png having 1 channel label.shape= (M, N).
    """


    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    # Ensure the output directories exist
    
    count = 0

    images_fname = [f for f in os.listdir(img_in_dir)]
    labels_fname = [f for f in os.listdir(label_in_dir)]

    if(len(images_fname) != len(labels_fname)):
        raise ValueError(f"Number of images and labels do not match. {len(images_fname)} for images",
                          f"{len(labels_fname)} for labels")

    try:
        for f in images_fname:
            fname_i = f"{count}.png" 
            fname_l = f"{count}.png"

            img_path = os.path.join(img_in_dir, f)
            label_path = os.path.join(label_in_dir, os.path.splitext(f)[0] + "_GT.png")

            label = io.imread(label_path)
            img = io.imread(img_path)
            
            #re-orient the image and label if they are not in the same orientation
            if img.shape[0] != label.shape[0] or img.shape[1] != label.shape[1]:
                print(f"image {img_path} and label {label_path} have different shapes, reshaping label to match image shape.")
                print("close window to continue...")
                label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
                plt.imshow(np.concatenate((img, label), axis=1))
                plt.show()

            #ensure the label is binary and has 1 channel i.e. 2 dimensions
            label = np.round(label/255).astype(np.uint8)[:,:,0]*255 #multiply 255 to display mask on png also

            img_out_path = os.path.join(img_out_dir,prefix, fname_i)
            label_out_path = os.path.join(label_out_dir,prefix, fname_l)

            shutil.copy(img_path, img_out_path)
            io.imsave(label_out_path, label, check_contrast=False)
            print(f"Saved {img_path} to {img_out_path}")
            print(f"Saved {label_path} to {label_out_path}")
            print(img.shape, label.shape)
            count += 1
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"Formatted {count} images and labels.")
        
    
if __name__ == "__main__":
    img_in_dir = "/home/user/henryshum0/Pytorch-UNet/Crack/images"
    label_in_dir = "/home/user/henryshum0/Pytorch-UNet/Crack/gt"
    img_out_dir = "/home/user/henryshum0/Pytorch-UNet/test/imgs"
    label_out_dir = "/home/user/henryshum0/Pytorch-UNet/test/masks"
    # prefix = "formatted"

    formatting_data(img_in_dir, label_in_dir, img_out_dir, label_out_dir)