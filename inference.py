


from torchvision.io.image import read_image
from torchvision.models.detection import *
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import cv2
from enhance import *
import torch
# from model import Autoencoder
# from dataset import Ex_dataset





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
                        width=3, font_size=15)
    im = to_pil_image(box.detach())
    im.show()
    
model = Autoencoder()
checkpoint_path = "Trained_model/model.h5"
model.load_weights(checkpoint_path)

img_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark\Car\2015_03008.png"
img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
print(img.shape)
# img_enhance = enhance(img, "linear_gray_transform")
img_enhance = enhance(img, "Autoencoder", model)
# img_enhance = enhance(img, "gamma_transform")
enhanced_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\image_test\enhanced.png"
cv2.imwrite(enhanced_path, img_enhance)
infer_one_image("fasterRCNN_restnet",
                img_path)
infer_one_image("fasterRCNN_restnet",
                enhanced_path)



