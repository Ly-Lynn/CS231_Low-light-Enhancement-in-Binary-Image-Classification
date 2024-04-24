# def inference(testDf, testDir, model, device='cpu', numImgs = 2, score_thres = 0.8):
    
#     """
#     """
    
    
#     model.to(device)
#     samples = random.sample(testDf.image_id.unique().tolist(), numImgs)
    
#     # Groundtruth bboxes
#     gt_Bboxes = []
#     for sample in samples:
#         rows = testDf[testDf['image_id'] == sample]
#         sample_gt_bboxes = []
#         for idx, row in rows.iterrows():
#             xmin, ymin, xmax, ymax = row['x_min'], row['y_min'], row['x_max'], row['y_max']
#             sample_gt_bboxes.append([xmin, ymin, xmax, ymax])
#         gt_Bboxes.append(sample_gt_bboxes)
        
#     inf_list = [read_image(os.path.join(testDir, img)) for img in samples]
    
#     # Predict bboxes
#     inf_float = [(img.float() / 255.0).to(device) for img in inf_list]
#     model.eval()
#     outputs = model(inf_float)
    
#     outImgs = []
#     #draw bboxes
#     for idx, img in enumerate(inf_list):
#         gtImg = draw_bounding_boxes(img, torch.tensor(gt_Bboxes[idx], dtype=torch.float32), colors='green', width=1)
#         predImg = draw_bounding_boxes(gtImg, boxes=outputs[idx]['boxes'][outputs[idx]['scores'] > score_thres], colors='red', width=1)
#         outImgs.append(predImg)
    
#     showImgs(make_grid(outImgs))


# # inference(testDf = testData, model = model, testDir = '/kaggle/input/kidney-stone-images/test/images')


from torchvision.io.image import read_image
from torchvision.models.detection import *
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from dataset import Ex_dataset


def infer_one_image(model_name, image_path, box_score_thresh=0.9):
    if model_name == "fasterRCNN_restnet":
        # model loading
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=box_score_thresh)
 
    
    elif model_name == "FCOS_ResNet50_FPN_Weights":   
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT
        model = fcos_resnet50_fpn(weights=weights, box_score_thresh=box_score_thresh)
    
    
    
    # ..................
    
    model.eval()

    # image loading and transform
    preprocess = weights.transforms()
    img = read_image(image_path) 
    batch = [preprocess(img)]
    
    # prediciton
    predictions = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in predictions["labels"]]
    print(predictions)
    box = draw_bounding_boxes(img, boxes=predictions["boxes"],
                        labels=labels,
                        font="ToThePointRegular-n9y4.ttf",
                        width=5, font_size=100)
    im = to_pil_image(box.detach())
    im.show()

        
infer_one_image("fasterRCNN_restnet", 
                r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\image_test\dog_people_1.jpg")


