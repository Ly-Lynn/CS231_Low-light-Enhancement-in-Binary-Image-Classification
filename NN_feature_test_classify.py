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
from dataset import draw_bounding_boxes, IOU, accuracy
import cv2
from enhance import enhance



def test_classify(model, test_loader, criterion):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    with torch.no_grad():

        for (imgs, labels, bbs, img_path) in tqdm(test_loader):
                imgs, labels = imgs.cuda(), labels.cuda()        
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss_meter.update(loss.item(), imgs.shape[0])
                acc = accuracy(torch.round(outputs), labels)
                print("\n", acc)
                acc_meter.update(acc, imgs.shape[0])

                

        print("Test loss ", loss_meter.avg)
        print("Accuracy ", acc_meter.avg)

def test(img, model):
    with torch.no_grad():
        img_ = transform(img).unsqueeze(0)
        
        output = model(img_.cuda()).round()
        
        label = "Dog" if output.item() == 0 else "Cat"
        
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
        plt.show()
        

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # ------------------ Load_dataset
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)), ######
        transforms.ToTensor()
    ])
    # test_dataset = ExDark_pytorch(annotations_file="Test.txt", transform=transform)



    # ---------------- Model
    model = Classification().eval().cuda()

    pretrained_path = r"Trained_model/best_classify_NN.pth"
    model.load_state_dict(torch.load(pretrained_path))
    criterion = nn.BCELoss()  

    # ham cal IOU score
    test_dataset = ExDark_pytorch("Splits/Test.txt", transform)
    # test_dataset = ExDark_pytorch(annotations_file="Splits/Test.txt", 
    #                                transform=transform, 
    #                                enhance_type="log_transform") # 0.41 iou
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    test_classify(model, test_loader, criterion)
    
    
    

        


    # image_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark\Dog\2015_05610.jpg"
    # img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    # # img = enhance(img, "log_transform")
    # test(img, model)
main()