import random
import os
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, make_grid
from torchvision.io import read_image

from visualize import showImgs

def inference(testDf, testDir, model, device='cpu', numImgs = 2, score_thres = 0.8):
    model.to(device)
    samples = random.sample(testDf.image_id.unique().tolist(), numImgs)
    
    # Groundtruth bboxes
    gt_Bboxes = []
    for sample in samples:
        rows = testDf[testDf['image_id'] == sample]
        sample_gt_bboxes = []
        for idx, row in rows.iterrows():
            xmin, ymin, xmax, ymax = row['x_min'], row['y_min'], row['x_max'], row['y_max']
            sample_gt_bboxes.append([xmin, ymin, xmax, ymax])
        gt_Bboxes.append(sample_gt_bboxes)
        
    inf_list = [read_image(os.path.join(testDir, img)) for img in samples]
    
    # Predict bboxes
    inf_float = [(img.float() / 255.0).to(device) for img in inf_list]
    model.eval()
    outputs = model(inf_float)
    
    outImgs = []
    #draw bboxes
    for idx, img in enumerate(inf_list):
        gtImg = draw_bounding_boxes(img, torch.tensor(gt_Bboxes[idx], dtype=torch.float32), colors='green', width=1)
        predImg = draw_bounding_boxes(gtImg, boxes=outputs[idx]['boxes'][outputs[idx]['scores'] > score_thres], colors='red', width=1)
        outImgs.append(predImg)
    
    showImgs(make_grid(outImgs))


# inference(testDf = testData, model = model, testDir = '/kaggle/input/kidney-stone-images/test/images')