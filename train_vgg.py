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


def train_localize(model, train_loader, val_loader, criterion, optimizer, epochs):
    loss_meter = AverageMeter()
    indx = 0
    best_loss = None
    for epoch in tqdm(range(epochs)):
        loss_meter.reset()
        
        for (imgs, labels, bbs, img_path) in tqdm(train_loader):
            try:
                indx += 1
                optimizer.zero_grad()
                imgs, bbs = imgs.cuda(), bbs.cuda()        
                position_outputs = model(imgs)

                loss = criterion(position_outputs, bbs) 
                loss_meter.update(loss.item(), imgs.shape[0])
                loss.backward()
                optimizer.step()
            except:
                print(img_path)     
        print(f"Train Loss: {loss_meter.avg}")
        loss_meter.reset()
        
    with torch.no_grad():
            
        for (imgs, labels, bbs, img_path) in tqdm(val_loader):
            imgs, bbs = imgs.cuda(), bbs.cuda()        
            position_outputs = model(imgs)
            loss = criterion(position_outputs, bbs) 
            loss_meter.update(loss.item(), imgs.shape[0])
        if not best_loss or loss_meter.avg < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "best_localize_log_enhance.pth")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# ------------------ Load_dataset
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)), ######
    transforms.ToTensor()
])
train_dataset = ExDark_pytorch(annotations_file="Train.txt", 
                               transform=transform, 
                               enhance="log_transform")
test_dataset = ExDark_pytorch(annotations_file="Test.txt", 
                               transform=transform, 
                               enhance="log_transform")
# val_dataset = ExDark_pytorch("Test.txt", transform)[:50]
# test_dataset = ExDark_pytorch("Test.txt", transform)[50:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)



# ---------------- Model
model = VGG().train().cuda()
pretrained_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\best_localize.pth"
model.load_state_dict(torch.load(pretrained_path))
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# ham cal IOU score
epochs = 200


train_localize(model, train_loader, test_loader, criterion, optimizer, epochs)