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
from model import Classification
import torch.nn as nn


os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def train(enhance=None):
    
    W, H = 128, 128 # image to resize
    
    # ------------------ Load_dataset -------------------------------
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((W, H)),
        transforms.ToTensor()
    ]) # transform data
    
    
    train_dataset = ExDark_pytorch(annotations_file="Train_1.txt", 
                                transform=transform, 
                                enhance_type=enhance)
    test_dataset = ExDark_pytorch(annotations_file="Test_1.txt", 
                                transform=transform, 
                                enhance_type=enhance)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # -------------------- Load model ----------------------
    model = Classification().cuda().train()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    epochs = 200

    loss_meter = AverageMeter() # loss tracking
    best_loss = None # loss for validation
    
    for epoch in tqdm(range(epochs)):
        
        loss_meter.reset()
        model.train()
        
        for (imgs, labels, bbs, img_path) in tqdm(train_loader):
            try:
                optimizer.zero_grad()
                imgs, labels = imgs.cuda(), labels.cuda()   
                outputs = model(imgs)
                loss = criterion(outputs, labels) 
                loss_meter.update(loss.item(), imgs.shape[0])
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(img_path)

        print(f"Train Loss: {loss_meter.avg}")
        loss_meter.reset()
        
        # validation
        # with torch.no_grad():
        #     model.eval()
        #     try:
        #         for (imgs, labels, bbs, img_path) in tqdm(test_loader):
        #             imgs, labels = imgs.cuda(), labels.cuda()        
        #             outputs = model(imgs)
        #             loss = criterion(outputs, labels)
        #             loss_meter.update(loss.item(), imgs.shape[0])
        #         if not best_loss or loss_meter.avg < best_loss:
        #             best_loss = loss_meter.avg
        #             torch.save(model.state_dict(), f"best_classify_{enhance}_.pth")
        #     except Exception as e:
        #         print(img_path)
        #         break

train()
# train()
# train("PCA", enhance=None)

# orignal: 0.648
# linear gray: 0.674
# log : 0.73
# gamma: 0.59