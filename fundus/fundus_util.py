import random
import os
import glob
import numpy as np
import cv2

from tf_unet.image_util import BaseDataProvider

class fundusDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    def __init__(self, file_type, a_min=None, a_max=None, image_path="./image/", mask_path='./mask/', shuffle_data=True, n_class = 2):
        super(fundusDataProvider, self).__init__(a_min, a_max)
        self.file_type = file_type
        self.image_path = image_path
        self.mask_path = mask_path
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        
        self.image_files, self.mask_files = self._find_files()
        
        assert len(self.image_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.image_files))

        if self.shuffle_data:
            self._shuffle_file()
            
        img = self._load_file(self.image_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_files(self):
        image_patterns = os.path.join(self.image_path,"*.{}".format(self.file_type))
        image_files = glob.glob(image_patterns)
        
        mask_patterns = os.path.join(self.mask_path,"*.{}".format(self.file_type))
        mask_files = glob.glob(mask_patterns)
        
        image_set = set(os.path.split(filepath)[1] for filepath in image_files)
        mask_set = set(os.path.split(filepath)[1] for filepath in mask_files)
        exist_set = image_set & mask_set 

        exist_image_files = [filepath for filepath in image_files if os.path.split(filepath)[1] in exist_set]
        exist_mask_files = [filepath for filepath in mask_files if os.path.split(filepath)[1] in exist_set]
        
        return exist_image_files, exist_mask_files
    
    def _load_file(self, path, dtype=np.float32):
        if dtype == np.bool:
            img = cv2.imread(path,0)
            img = img.astype(dtype)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(dtype)
        return img

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.image_files):
            self.file_idx = 0 
            if self.shuffle_data:
                self._shuffle_file()
    
    def _shuffle_file(self):
        zip_ = list(zip(self.image_files,self.mask_files))
        random.shuffle(zip_)
        self.image_files, self.mask_files = zip(*zip_)
    
    def _next_data(self):
        self._cylce_file()
        image_name = self.image_files[self.file_idx]
        mask_name = self.mask_files[self.file_idx]
        
        
        if not os.path.splitext(os.path.split(image_name)[1])[0] == os.path.splitext(os.path.split(mask_name)[1])[0]:
            raise ValueError("unsync image and mask {} - {}".format())
            
        img = self._load_file(image_name, np.float32)
        label = self._load_file(mask_name, np.bool)
    
        return img,label