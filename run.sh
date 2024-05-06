# training model
python train.py --train_model classification --pretrained Trained_model/best_classify_NN.pth --train_annotator Splits/Train.txt --test_annotator Splits/Test.txt --enhanced_type log_transform
# testing model
python test.py --test_model classification --pretrained Trained_model/best_classify_NN.pth --test_annotator Splits/Test.txt --enhanced_type log_transform 