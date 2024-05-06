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
from dataset import draw_bounding_boxes, IOU, accuracy
import cv2
import argparse



parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gpu', default="0", type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--test_model', default="classification", type=str,
                    help='classification or localization')
parser.add_argument("--pretrained", default=None, type=str, 
                    help="path to pretrained classification or localization model")
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



def test_classify(test_loader):
    
    model = Classification().cuda().train()
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with torch.no_grad():
        for (imgs, labels, bbs, img_path) in tqdm(test_loader):
                imgs, labels = imgs.cuda(), labels.cuda()        
                outputs = model(imgs)
                acc = accuracy(torch.round(outputs), labels)
                print("\n", acc)
                acc_meter.update(acc, imgs.shape[0])

        print("Accuracy Score", acc_meter.avg)
        
def test_localize(test_loader):
    model = Localization().train().cuda()
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))

    iou_meter = AverageMeter()
    
    with torch.no_grad():
        
        for (imgs, labels, bbs, img_path) in tqdm(test_loader):
            imgs, bbs = imgs.cuda(), bbs.cuda()
            position_outputs = model(imgs)
            iouscore = IOU(bbs, position_outputs).mean()
            iou_meter.update(iouscore.item(), imgs.shape[0])

        print("IOU score", iou_meter.avg)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor()
    ]) 
    
    # --------------------- Dataloader -------------------------------
    test_dataset = ExDark_pytorch(annotations_file=args.test_annotator, 
                                transform=transform, 
                                enhance_type=args.enhanced_type)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    
    if args.test_model == "classification":
        test_classify(test_loader)
    elif args.test_model == "localization":
        test_localize(test_loader)

if __name__ == "__main__":
    main()