# IDOA2021
IDOA2021 Stage 1 solution


Instruction:

1. Please check your path for data. BASE_PATH should be same as your path for data.
2. First run the "prepare_labels.py" file. There will be 3 files saved on your computer (train_label.csv, test_imgs.npy and train_imgs.npy)
3. Second run the "train_models_for_pseudo_labeling.py" file. Please create "models", and "oof"  folders on your computer for saving trained models and their results. We chose sample from test set as a validation set and other images are training set. Note that the images size should be (96, 96) for image augmentation. Please change image size on "Dataset.py" file. 
4. Third run "prediction.py" file. Note that the "Pseudo_labeling" and "Only_private_set" parameters must be True ! This file print "submission_private.csv" file on your computer. 
5. Fourth run the "pseudo_labeling.py" file. Based on the predicted results ("submission_private.csv" file), it will perform a pseudo labeling. It will save some of test images into training set. This file print "pseudo_label.csv" file on your computer.
6. Sixth run the "train_final_models.py" file. Please create "final_models"  folder on your computer for saving trained final models. Due to the time limit in track 2, we use 3 cross validation method. Note that the images size should be (64, 64) for image augmentation. Please change image size on "Dataset.py" file.
7. Finally, run "prediction.py" file. Note that the "Pseudo_labeling" and "Only_private_set" parameters must be False! This file print "submission.csv" file on your computer. It should be our final predicted file. 
8. For track 2, please change BASE_PATH parameter for your system. It should be "tests". 
