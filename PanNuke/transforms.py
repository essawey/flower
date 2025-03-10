# pannuke/transforms.py
import os
import albumentations as A
from pathlib import Path
from torchvision import transforms

from PIL import Image
import cv2
import math
import numpy as np

def get_data_augmentation(means, stds):
    """Returns data augmentation pipelines for training, validation, and testing."""
    
    data_augmentation = {
        'Train': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.Normalize(mean=means, std=stds),
        ]),
        'Validation': A.Compose([
            A.Normalize(mean=means, std=stds),
        ]),
        'Test': A.Compose([
            A.Normalize(mean=means, std=stds),
        ]),
    }

    return data_augmentation


from hydra import initialize, compose
initialize(config_path="./conf")
cfg = compose(config_name="base")

def create_patches(image_dir, target_dir, patch_size = cfg.patch_size):
    for path, _, _ in sorted(os.walk(image_dir)):
        relative_path = os.path.relpath(path, image_dir)
        target_path = Path(target_dir) / relative_path
        
        target_path.mkdir(parents=False, exist_ok=True)

        images_index, masks_index = 0, 0

        images = sorted(os.listdir(path))
        for image_name in images:
            if image_name.endswith(".png") or image_name.endswith(".npy"):
                if image_name.endswith(".npy"):
                    image = np.load(os.path.join(path, image_name))
                    image = np.transpose(image, (1, 2, 0)) # (h, w, c)
                    image = np.argmax(image, axis=-1) # (h, w)
                    image = image.astype(np.uint8)
                else:
                    image = cv2.imread(os.path.join(path, image_name))
                size_X, size_Y = math.ceil(image.shape[1]/patch_size), math.ceil(image.shape[0]/patch_size)
                pad_X, pad_Y = (patch_size * size_X - image.shape[1]) / (size_X - 1), (patch_size * size_Y - image.shape[0]) / (size_Y - 1)
                image = Image.fromarray(image)
                top = 0
                for y in range(size_Y):
                    left = 0
                    for x in range(size_X):
                        crop_image = transforms.functional.crop(image, top, left, patch_size, patch_size)
                        crop_image = np.array(crop_image)
                        if image_name.endswith('.png'):
                            patch_name = f"{Path(image_name).stem}_patch{images_index}.png"
                            cv2.imwrite(str(target_path / patch_name), crop_image)
                            images_index += 1
                        else:
                            patch_name = f"{Path(image_name).stem}_patch{masks_index}.npy"
                            np.save(str(target_path / patch_name), crop_image)
                            masks_index += 1
                        left = left + patch_size - pad_X
                    top = top + patch_size - pad_Y