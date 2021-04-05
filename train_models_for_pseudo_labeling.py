import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
import os
from Dataset import *
from tqdm import tqdm
from torchvision import transforms
from torchvision import models
from models import *
from sklearn.model_selection import KFold, GroupKFold
from MobileNet import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, mean_absolute_error
import torch.nn.functional as F
import random
from sklearn.utils import shuffle

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = 'idao_dataset/track_1_train_public/train/'
test_imgs = np.load('test_imgs.npy')
train_imgs = np.load('train_imgs.npy')

train_label = pd.read_csv('train_label.csv')
train_label = shuffle(train_label)
train_label = train_label.reset_index(drop=True)

for task_type in ['classification', 'regression']:

    if task_type=='classification':
        mode = 'max'
        group = train_label['reg_target'].values
        Nfold = 6
    else:
        mode = 'min'
        group = train_label['cl_target'].values + 1 + train_label['reg_target'].values
        Nfold = 12


    kf = GroupKFold(n_splits=Nfold)
    fold = 0

    index = np.arange(0, train_label.shape[0])
    for train_index, test_index in kf.split(train_label.index, groups=group):
        fold = fold + 1

        if fold>1:
            print("There is no enough test set")
            break

        else:

            train_all, test_all = train_label.loc[train_index, 'path'].reset_index(drop=True).values, \
                                  train_label.loc[test_index, 'path'].reset_index(drop=True).values

            #please check image size in Dataset.py file. Image size should be equal to (96, 96) for pseudo labeling model.
            train_set = IDOADataset(imgs=train_imgs, base_path=BASE_PATH, transforms=train_transform, train=True)
            train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)

            # please check image size in Dataset.py file. Image size should be equal to (96, 96) for pseudo labeling model.
            test_set = IDOADataset(imgs=test_imgs, base_path=BASE_PATH, transforms=test_transform, train=True)
            test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)

            epochs = 50
            pre_model ='local'

            if pre_model == 'vgg':
                arch =  models.vgg16_bn(pretrained=True)
            elif pre_model == 'mobilenet':
                arch = models.mobilenet_v2(pretrained=True)
            elif pre_model=='local':
                arch = mobilenet_v2(pretrained=False)

            model = Net(arch=arch, out_dim=1)  # New model for each fold
            model = model.to(device)
            optim = torch.optim.Adam(model.parameters(), lr=0.0001)
            scheduler = ReduceLROnPlateau(optimizer=optim, mode=mode, patience=3, verbose=True, factor=0.1)

            if task_type=='classification':
                criterion = nn.BCEWithLogitsLoss()
                best_score = 0
            else:
                criterion = nn.L1Loss()
                best_score = np.inf

            counter = 0
            for epoch in range(epochs):
                train_preds = []
                y_true = []
                train_preds_cl = []
                y_true_cl = []
                epoch_loss = 0
                for img, (reg_target, class_target) in tqdm(train_loader):
                    out = model(img.to(device))

                    if task_type=='classification':
                        loss = criterion(out, class_target.unsqueeze(1).to(device))
                        train_preds += list(out.sigmoid().detach().cpu().numpy().flatten())
                        y_true += list(class_target.detach().cpu().numpy().flatten())
                    else:
                        loss = criterion(out, reg_target.unsqueeze(1).to(device))
                        train_preds += list(out.detach().cpu().numpy().flatten())
                        y_true += list(reg_target.detach().cpu().numpy().flatten())


                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    epoch_loss += loss.item()

                if task_type == 'classification':
                    metric = roc_auc_score(y_true, train_preds)
                else:
                    metric = mean_absolute_error(y_true, train_preds)

                print(
                    'Epoch {:03}: | Loss: {:.3f} | Train metric: {:.3f}'.format(
                        epoch + 1,
                        epoch_loss,
                        metric))

                val_preds = []
                y_true = []
                val_preds_cl = []
                y_true_cl = []
                model.eval()
                with torch.no_grad():
                    for img, (reg_target, class_target) in tqdm(test_loader):

                        out = model(img.to(device))

                        if task_type == 'classification':
                            val_preds += list(out.sigmoid().detach().cpu().numpy().flatten())
                            y_true += list(class_target.detach().cpu().numpy().flatten())
                        else:
                            val_preds += list(out.detach().cpu().numpy().flatten())
                            y_true += list(reg_target.detach().cpu().numpy().flatten())

                    if task_type == 'classification':
                        metric = roc_auc_score(y_true, val_preds)
                    else:

                        metric = mean_absolute_error(y_true, val_preds)

                    print(
                        'Epoch {:03}: | Loss: {:.3f} | Val metric: {:.3f}'.format(
                            epoch + 1,
                            epoch_loss,
                            metric))

                    scheduler.step(metric)

                    if task_type=='classification':
                        if best_score < metric:
                            counter=0
                            best_score = metric
                            torch.save(model, f'models/model_{task_type}_{fold}.pth')
                            np.save(f'oof/reg_pred_{task_type}_{fold}.npy', np.asarray(val_preds))
                            np.save(f'oof/reg_target_{task_type}_{fold}.npy', np.asarray(y_true))
                        else:
                            counter+=1
                    else:
                        if best_score > metric:
                            print('model updated...')
                            counter = 0
                            best_score = metric
                            torch.save(model, f'models/model_{task_type}_{fold}.pth')
                            np.save(f'oof/reg_pred_{task_type}_{fold}.npy', np.asarray(val_preds))
                            np.save(f'oof/reg_target_{task_type}_{fold}.npy', np.asarray(y_true))
                        else:
                            counter += 1

                    if counter >= 5:
                        print("Early stopping")
                        break



