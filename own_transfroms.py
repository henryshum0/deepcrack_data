from torchvision import transforms
from numpy.random import rand



def random_crop(image, mask):
    """
    Randomly crop the image and mask to the same size.
    """
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
    image = transforms.functional.crop(image, i, j, h, w)
    mask = transforms.functional.crop(mask, i, j, h, w)
    return image, mask

def random_horizontal_flip(image, mask):
    """
    Randomly flip the image and mask horizontally.
    """
    if rand() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)
    return image, mask

def normalize(image, mask):
    """
    Normalize the image with the given mean and standard deviation.
    normalization on our data may actually cause the cracks in image to be less visible
    """
    image = transforms.functional.normalize(image, mean=[.45, .45, .45], std=[0.229, 0.224, 0.225])
    return image, mask

def random_affine(image, mask):
    """
    Apply random affine transformation to the image and mask.
    """
    angle = transforms.RandomAffine.get_params(degrees=[-10, 10], translate=[0.1, .5], scale_ranges=[0.7, 1.5], shears=None, img_size=image.shape)
    image = transforms.functional.affine(image, angle=angle[0], translate=angle[1], scale=angle[2], shear=angle[3])
    mask = transforms.functional.affine(mask, angle=angle[0], translate=angle[1], scale=angle[2], shear=angle[3])
    return image, mask

def random_rotation(image, mask):
    """
    Randomly rotate the image and mask.
    """
    angle = transforms.RandomRotation.get_params([-30, 30])
    image = transforms.functional.rotate(image, angle)
    mask = transforms.functional.rotate(mask, angle)
    return image, mask

def random_color_jitter(image, mask):
    """
    Randomly change the brightness, contrast, saturation, and hue of the image.
    """
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    image = color_jitter(image)
    return image, mask

def random_gaussian_blur(image, mask):
    """
    Randomly apply Gaussian blur to the image.
    """
    if rand() > 0.5:
        image = transforms.functional.gaussian_blur(image, kernel_size=(5, 5), sigma=(0.1, 2.0))
    return image, mask




