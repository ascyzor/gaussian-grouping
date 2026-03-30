import os
from os import path
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np

_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class SimpleVideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    This simple version:
    1. Does not load the mask/json
    2. Does not normalize the input
    3. Does not resize

    Supports both flat directories (all images directly inside image_dir) and
    nested layouts where images live in one level of subdirectories
    (e.g. image_dir/pano_camera0/*.jpg).
    """
    def __init__(
        self,
        image_dir,
    ):
        """
        image_dir - points to a directory of jpg/png/… images, or a directory
                    whose immediate subdirectories contain such images.
        """
        self.image_dir = image_dir

        # Collect all image file paths recursively, then sort so that the
        # temporal order is consistent regardless of the directory layout.
        image_paths = []
        for dirpath, _dirnames, filenames in os.walk(image_dir):
            for fname in filenames:
                if path.splitext(fname)[1].lower() in _IMAGE_EXTENSIONS:
                    image_paths.append(path.join(dirpath, fname))
        self.frames = sorted(image_paths)

    def __getitem__(self, idx):
        im_path = self.frames[idx]
        img = Image.open(im_path).convert('RGB')
        img = np.array(img)

        return img, im_path
    
    def __len__(self):
        return len(self.frames)
    

def no_collate(x):
    return x