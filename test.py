import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import cv2 as cv
from torch.utils.data import DataLoader
from torch import optim
import torchvision.models as models
from dataset import AverageMeter, ExDark_pytorch
from model import VGG
import torch.nn as nn
from dataset import draw_bounding_boxes, IOU
import cv2

def test_localize(model, test_loader, criterion):
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    with torch.no_grad():
        
        for (imgs, labels, bbs, img_path) in tqdm(test_loader):
            imgs, bbs = imgs.cuda(), bbs.cuda()
            print(bbs.shape)
            position_outputs = model(imgs)
            print(position_outputs.shape)
            loss = criterion(position_outputs, bbs) 
            loss_meter.update(loss.item(), imgs.shape[0])
            iouscore = IOU(bbs, position_outputs).mean()
            iou_meter.update(iouscore.item(), imgs.shape[0])

            break
        print("Test loss", loss_meter.avg)
        print("IOU score", iou_meter.avg)

def vissulize_test(img, model):
    imgs = transform(img)
    w, h = img.shape[0], img.shape[1]
    bb_preds = model(imgs.unsqueeze(0).cuda())
    bb_preds[0][0] *= w / 128
    bb_preds[0][1] *= h / 128
    bb_preds[0][2] *= w / 128
    bb_preds[0][3] *= h / 128

    img_draw = draw_bounding_boxes(img, bb_preds.squeeze())
    cv2.imshow("Img_draw", img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# ------------------ Load_dataset
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)), ######
    transforms.ToTensor()
])
# test_dataset = ExDark_pytorch(annotations_file="Test.txt", transform=transform)



# ---------------- Model
model = VGG().train().cuda()

pretrained_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\best_localize_log_enhance.pth"
model.load_state_dict(torch.load(pretrained_path))
criterion = nn.MSELoss()  

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)), ######
    transforms.ToTensor()
])

# ham cal IOU score
# test_dataset = ExDark_pytorch("Test.txt", transform)
test_dataset = ExDark_pytorch(annotations_file="Test.txt", 
                               transform=transform, 
                               enhance="log_transform") # 0.41 iou
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
test_localize(model, test_loader, criterion)

    


# img_path = ""
# image_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark\Dog\2015_05599.jpg"
# img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
# vissulize_test(img, model)