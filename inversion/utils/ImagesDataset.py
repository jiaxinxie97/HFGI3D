import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils.data_utils import make_dataset


class ImagesDataset(Dataset):

    def __init__(self, source_root,name, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root,name))
        self.source_transform = source_transform
      

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, c_path, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        from_c = np.load(c_path)
        from_c = np.array(from_c,dtype=np.float32)
        if self.source_transform:
            from_im = self.source_transform(from_im)
        
        return fname, from_im,from_c
