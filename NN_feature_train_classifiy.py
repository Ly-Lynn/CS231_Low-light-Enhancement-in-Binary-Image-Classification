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
from dataset import AverageMeter, ExDark_pytorch, ExDark_ML_base
from model import Classification, ML_base_model
import torch.nn as nn


os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def train(model_name, enhance=None):
    

    if model_name == "NN":
    
        # ------------------ Load_dataset
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        train_dataset = ExDark_pytorch(annotations_file="Train.txt", 
                                    transform=transform, 
                                    enhance_type=enhance)
        test_dataset = ExDark_pytorch(annotations_file="Test.txt", 
                                    transform=transform, 
                                    enhance_type=enhance)
        model = Classification().cuda()
    
    
    elif model_name == "PCA":
        
        transform = transforms.ToTensor()
        
        train_dataset = ExDark_ML_base(annotations_file="Train.txt", 
                                    enhance=enhance,
                                    transform=transform, 
                                    feature_pretrained="PCA_None.pkl")
        test_dataset = ExDark_ML_base(annotations_file="Test.txt", 
                                    enhance=enhance, 
                                    transform=transform, 
                                    feature_pretrained="PCA_None.pkl")
        model = ML_base_model().cuda()


    elif model_name == "":
        print()
        ################################################################

    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


    # ---------------- Model
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    epochs = 200

    loss_meter = AverageMeter()
    indx = 0
    best_loss = None
    for epoch in tqdm(range(epochs)):
        loss_meter.reset()
        model.train()
        for (imgs, labels, bbs, img_path) in tqdm(train_loader):
            try:
                indx += 1
                optimizer.zero_grad()
                imgs, labels = imgs.cuda(), labels.cuda()   
                outputs = model(imgs)
                loss = criterion(outputs, labels) 
                loss_meter.update(loss.item(), imgs.shape[0])
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(e)
                break

        print(f"Train Loss: {loss_meter.avg}")
        loss_meter.reset()
        
        with torch.no_grad():
            model.eval()
            for (imgs, labels, bbs, img_path) in tqdm(test_loader):
                imgs, labels = imgs.cuda(), labels.cuda()        
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss_meter.update(loss.item(), imgs.shape[0])
            if not best_loss or loss_meter.avg < best_loss:
                best_loss = loss_meter.avg
                torch.save(model.state_dict(), f"best_classify_{model_name}_{enhance}.pth")
                # torch.save(model.state_dict(), f"best_classify_test.pth")

# train("NN", enhance=None)
train("PCA", enhance=None)