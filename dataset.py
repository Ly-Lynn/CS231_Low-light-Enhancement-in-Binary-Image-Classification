
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import *
from torchvision import transforms
from enhance import enhance
import cv2
import os
import joblib

def accuracy(output, target):    
    correct = output.eq(target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy
    

def IOU(bbox1, bbox2):
    x_min1, y_min1, x_max1, y_max1 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    x_min2, y_min2, x_max2, y_max2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    x_min_inter = torch.max(x_min1, x_min2)
    y_min_inter = torch.max(y_min1, y_min2)
    x_max_inter = torch.min(x_max1, x_max2)
    y_max_inter = torch.min(y_max1, y_max2)

    inter_area = torch.clamp(x_max_inter - x_min_inter, min=0) * torch.clamp(y_max_inter - y_min_inter, min=0)

    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou


def draw_bounding_boxes(image, bbs):
    x1, y1, x2, y2 = bbs
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ bounding box

    return image


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class ExDark_pytorch(Dataset):
    def __init__(self, annotations_file, 
                 transform, 
                 enhance_type=None,
                 anno_dir=r"D:/AI/CV/CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks/ExDark_Annno",
                 img_dir=r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark"):
        
        with open(annotations_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        
        self.lines = lines
        self.transform = transform
        self.enhance_type = enhance_type
        self.anno_dir = anno_dir
        self.img_dir = img_dir
    
    def __getitem__(self, index):
        [anno_path, label] = self.lines[index].split(", ")
        
        label = 0 if label == "Dog" else 1
        # label = torch.tensor(label)
        label = torch.tensor([label]).float()
        
        img_path = anno_path.replace(self.anno_dir, self.img_dir).replace(".txt", "")
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        
        
        with open(anno_path, "r") as f:
            f.readline()
            line = f.readline().split()
        [x_min, y_min, w_b, h_b] = [int(coordinate) for coordinate in line[1:5]]
        x_max, y_max = x_min + w_b, y_min + h_b
        
        w, h = img.shape[0], img.shape[1]
        x_ratio = w / 128
        y_ratio = h / 128
        x_min, y_min, x_max, y_max = int(x_min / x_ratio), int(y_min / y_ratio),  int(x_max / x_ratio), int(y_max / y_ratio)
        bb = torch.tensor([x_min, y_min, x_max, y_max])
        
        if self.enhance_type:
            img = enhance(img, self.enhance_type)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, bb.float(), img_path
            
    def __len__(self):
        return len(self.lines)
    
