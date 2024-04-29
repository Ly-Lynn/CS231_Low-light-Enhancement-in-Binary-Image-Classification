import os
import cv2
from sklearn.decomposition import PCA
import joblib
import numpy as np
from enhance import enhance

def load_imgs(annotations_file,
              enhance_type,
              anno_dir=r"D:/AI/CV/CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks/ExDark_Annno",
              img_dir=r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark"):
    with open(annotations_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
    
    imgs = []
        
    for line in lines:
        [anno_path, label] = line.split(", ")
        img_path = anno_path.replace(anno_dir, img_dir).replace(".txt", "")
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    
    if enhance_type:
        imgs = [enhance(img, enhance_type) for img in imgs]
    
    imgs = [cv2.resize(img, (128, 128)) for img in imgs]

    
    return imgs
        
    
def create_feature(type="PCA", enhance_type=None):
    imgs = load_imgs("Train.txt", enhance_type)
    imgs = np.array(imgs)
    
    if type == "PCA":
        flat_imgs = imgs.reshape(imgs.shape[0], -1)
        pca = PCA(n_components=700)  
        pca.fit(flat_imgs)
        
    model_path=f"{type}_{enhance_type}.pkl"
    joblib.dump(pca, model_path)

# def transform()

# pca = joblib.load("PCA_None.pkl")
# img_path = r"D:\AI\CV\CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks\ExDark\ExDark\Cup\2015_04425.jpg"
# img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (128, 128))
# img = img.reshape(1, -1)
# img_transform = pca.transform(img)
# print(img_transform.shape)
    
    
