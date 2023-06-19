import torch
import os
import glob
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
import numpy as np
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    # def __getitem__(self, index):
    #     """Return a data point and its metadata information.

    #     Parameters:
    #         index - - a random integer for data indexing

    #     Returns a dictionary that contains A, B, A_paths and B_paths
    #         A (tensor) - - an image in the input domain
    #         B (tensor) - - its corresponding image in the target domain
    #         A_paths (str) - - image paths
    #         B_paths (str) - - image paths (same as A_paths)
    #     """
    #     # read a image given a random integer index
    #     AB_path = self.AB_paths[index]
    #     AB = Image.open(AB_path).convert('RGB')
    #     # split AB image into A and B
    #     w, h = AB.size
    #     w2 = int(w / 2)
    #     A = AB.crop((0, 0, w2, h))
    #     B = AB.crop((w2, 0, w, h))

    #     # apply the same transform to both A and B
    #     transform_params = get_params(self.opt, A.size)
    #     A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
    #     B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

    #     A = A_transform(A)
    #     B = B_transform(B)

    #     return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
    
    def __getitem__(self, index):
        # get the base path for the current item
        base_path = self.AB_paths[index]

        # read the images and seed numbers
        A1 = Image.open(os.path.join(base_path, 'A1.png')).convert('RGB')
        A2 = Image.open(os.path.join(base_path, 'A2.png')).convert('RGB')
        A3 = Image.open(os.path.join(base_path, 'A3.png')).convert('RGB')
        B = Image.open(os.path.join(base_path, 'B.png')).convert('RGB')
        with open(os.path.join(base_path, 'seed.txt'), 'r') as f:
            seed = np.array([float(num) for num in f.read().split(',')])
        seed = torch.from_numpy(seed)

        # apply the same transform to all images
        transform_params = get_params(self.opt, A1.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        A1 = A_transform(A1)
        A2 = A_transform(A2)
        A3 = A_transform(A3)
        B = B_transform(B)

        return {'A1': A1, 'A2': A2, 'A3': A3, 'seed': seed, 'B': B, 'A_paths': base_path, 'B_paths': base_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
