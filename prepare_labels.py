import cv2
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import os

## train set analysis ###

#get labels for each image
BASE_PATH = 'idao_dataset/track_1_train_public/train/'
files = os.listdir(BASE_PATH)
print(files)
ALL = []
cl_target = []

for i, f in enumerate(files):
    imgs = os.listdir(BASE_PATH+f)
    print(np.zeros(len(imgs)) + i)
    cl = np.zeros(len(imgs)) + i
    cl_target.append(cl)
    ALL+=imgs

cl_target = np.concatenate(cl_target, axis=0)
cl_target = np.where(cl_target==0, 1, 0)

ALL=np.asarray(ALL)
target = []

for img in ALL:
    names = os.path.split(img)[-1].split("_")
    idx = [i for i, v in enumerate(names) if v == "keV"][0]
    reg_target = float(names[idx - 1])
    target.append(reg_target)

target = np.asarray(target)

df = pd.DataFrame()
df['cl_target'] = cl_target
df['reg_target'] = target
df['path'] = ALL

df.to_csv('train_label.csv', index=False)

cl = ['ER', 'NR']
lb = [1, 0]

train_imgs = []
test_imgs = []
for p, l in zip(cl, lb):
    temp = df.loc[df['cl_target']==l, :]
    for r in temp['reg_target'].unique():
        temp_img = temp.loc[temp['reg_target']==r, :]
        if temp_img.shape[0]<4:
            #imgs = []
            for img_path in temp_img['path']:
                test_imgs.append(img_path)
        else:
            for img_path in temp_img['path']:
                train_imgs.append(img_path)
                # img = cv2.imread(BASE_PATH + p + "/" + img_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # imgs.append(img)

#images for train and test sets.
print(len(train_imgs), len(test_imgs))
np.save('test_imgs.npy', np.asarray(test_imgs))
np.save('train_imgs.npy', np.asarray(train_imgs))

