
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import *

import os

class ExDark(Dataset):
    def __init__(self, imgs_root, anno_root, model_name="fasterRCNN_restnet"):
        self.imgs_root = imgs_root
        self.anno_root = anno_root        
        self.img_paths = [os.path.join(self.imgs_root, folder_name, file_name)
                          for folder_name in os.listdir(self.imgs_root)
                          for file_name in os.listdir(os.path.join(self.imgs_root, folder_name))]
        
        
        if model_name == "fasterRCNN_restnet":
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.transforms = weights.transforms()
        elif model_name == "....":
            print()
        
        # ..........
        

        

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.img_paths[idx]
        anno_path = img_path.replace(self.imgs_root, self.anno_root) + ".txt"
        with open(anno_path, 'r') as anno_file:
            anno_file.readline()
            lines = anno_file.readlines()
            lines = [line.split() for line in lines]
        
                
        img = read_image(img_path)
        img = self.transforms(img)
        
        labels = [line[0] for line in lines] # 
        boxes = torch.tensor([[int(coordinate) for coordinate in line[1:5]] for line in lines])
        # boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        
    

        return img, labels, boxes

    def __len__(self):
        return len(self.img_paths)
    
Ex_dataset = ExDark(r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark",
       r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark_Annno")

# a, b, c = dataset[0]
# print(a.shape)
# print(c.shape)
