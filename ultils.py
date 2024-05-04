import os
import json
import random

ANNNO_DIR = "D:/AI/CV/CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks/ExDark_Annno"
output_file = "analysis.json"

def extract_analysis_file(output_file = "analysis.json",
                          ANNNO_DIR = "D:/AI/CV/CS231_Low-light-Enhancement-in-Classical-Computer-Vision-Tasks/ExDark_Annno"
                          ):
    
    '''
    Extract the data of speicalize class with only one object in the image
    '''
    
    analysis = {}
    for class_name in os.listdir(ANNNO_DIR):
        class_path = os.path.join(ANNNO_DIR, class_name)
        if os.path.isdir(class_path):
            analysis[class_name] = []
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) == 2:
                        analysis[class_name].append(file_path)

    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=4)

def extract_data(analys_file, ratio=0.7):
    '''
    Splits data with ratio, default to class
    '''
    
    with open(analys_file, "r") as f:
        dic = json.load(f)
        dog_paths = dic["Dog"] # class
        cat_paths = dic["Cat"] # class
        
        # i = int(0.8 * len(dog_paths))
        # train_dog_paths = dog_paths[:i]
        # train_cat_paths = cat_paths[:i]
        # test_dog_paths = dog_paths[i:]
        # test_cat_paths = cat_paths[i:]
        
        train_dog_paths = random.sample(dog_paths, k=int(ratio * len(dog_paths)))
        train_cat_paths = random.sample(cat_paths, k=int(ratio * len(cat_paths)))
        
        test_dog_paths = [path for path in dog_paths if path not in train_dog_paths]
        test_cat_paths = [path for path in cat_paths if path not in train_cat_paths]
        
        print(len(train_dog_paths) + len(train_cat_paths))
        print(len(test_dog_paths) + len(test_cat_paths))

        
        with open("Train_2.txt", "w") as train_file:
            for path in train_dog_paths:
                train_file.write(f"{path}, Dog\n")
            for path in train_cat_paths:
                train_file.write(f"{path}, Cat\n")

        with open("Test_2.txt", "w") as test_file:
            for path in test_dog_paths:
                test_file.write(f"{path}, Dog\n")
            for path in test_cat_paths:
                test_file.write(f"{path}, Cat\n")
        
extract_data(output_file)
# Dog : 449
# Cat: 576
