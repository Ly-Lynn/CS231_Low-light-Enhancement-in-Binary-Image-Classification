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
from model import *
import torch.nn as nn
from enhance import *
import argparse


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gpu', default="0", type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--train_model', default="classification", type=str,
                    help='classification or localization')
parser.add_argument("--pretrained", default=None, type=str, 
                    help="path to pretrained classification or localization model")

parser.add_argument("--train_annotator", default=None, type=str, 
                    help="path to annotation csv file")
parser.add_argument("--test_annotator", default=None, type=str, 
                    help="path to annotation csv file")
parser.add_argument("--size", default=128, type=int, 
                    help="size = weight = height of the image to train")
parser.add_argument("--enhanced_type", default=None, type=str, 
                    help="low light enhanced technique to train [linear_gray_transform,log_transform, gamma_transform, HE, Autoencoder]")
parser.add_argument("--batch", default=32, type=int, 
                    help="batch size for training")
parser.add_argument("--epochs", default=200, type=int, 
                    help="Number of epochs for training")


args = parser.parse_args()



def train_classifier(train_loader, test_loader):
    # -------------------- Load model ----------------------
    model = Classification().cuda().train()
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    
    # ultil for train
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    loss_meter = AverageMeter()
    best_loss = None
    
    for epoch in tqdm(range(args.epochs)):
        
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
                print(e)
                print(img_path)

        print(f"Train Loss: {loss_meter.avg}")     
        loss_meter.reset()
   
        # if not best_loss or loss_meter.avg < best_loss:
        #     best_loss = loss_meter.avg
        #     torch.save(model.state_dict(), f"best_classify_{args.enhance_type}_.pth")


        # validation
        with torch.no_grad():
            model.eval()
            try:
                for (imgs, labels, bbs, img_path) in tqdm(test_loader):
                    imgs, labels = imgs.cuda(), labels.cuda()        
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss_meter.update(loss.item(), imgs.shape[0])
                if not best_loss or loss_meter.avg < best_loss:
                    best_loss = loss_meter.avg
                    torch.save(model.state_dict(), f"best_classify_{args.enhanced_type}_.pth")
            except Exception as e:
                print(e)


def train_localize(train_loader, test_loader):
    model = Localization().train().cuda()
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    loss_meter = AverageMeter()
    best_loss = None
    
    for epoch in tqdm(range(args.epochs)):
        
        loss_meter.reset()
        model.train()
        
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
                
            except Exception as e:
                print(e)
                print(img_path)

        print(f"Train Loss: {loss_meter.avg}")
        loss_meter.reset()
        
        # if not best_loss or loss_meter.avg < best_loss:
        #     best_loss = loss_meter.avg
        #     torch.save(model.state_dict(), f"best_classify_{args.enhance_type}_.pth")
       
       
        with torch.no_grad():
            model.eval()
            try:
                for (imgs, labels, bbs, img_path) in tqdm(test_loader):
                    imgs, bbs = imgs.cuda(), bbs.cuda()        
                    position_outputs = model(imgs)
                    loss = criterion(position_outputs, bbs)
                    loss_meter.update(loss.item(), imgs.shape[0])
                if not best_loss or loss_meter.avg < best_loss:
                    best_loss = loss_meter.avg
                    torch.save(model.state_dict(), f"best_localize_{args.enhanced_type}.pth")
            except Exception as e:
                print(img_path)
                print(e)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor()
    ]) 
    
    # --------------------- Dataloader -------------------------------
    train_dataset = ExDark_pytorch(annotations_file=args.train_annotator, 
                                transform=transform, 
                                enhance_type=args.enhanced_type)
    test_dataset = ExDark_pytorch(annotations_file=args.test_annotator, 
                                transform=transform, 
                                enhance_type=args.enhanced_type)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    
    if args.train_model == "classification":
        train_classifier(train_loader, test_loader)
    elif args.train_model == "localization":
        train_localize(train_loader, test_loader)


if __name__ == "__main__":
    main()
    
# orignal: 0.648
# linear gray: 0.674
# log : 0.73
# gamma: 0.59