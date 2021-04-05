import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score
import matplotlib.pylab as plt
import os
import pandas as pd
import cv2
from tqdm import tqdm

## train set analysis ### 
BASE_PATH =  'idao_dataset/track_1_train_public/'
df = pd.read_csv('train_label.csv')

for i in range(2):
    print('*'*50)
    temp = df.loc[df['cl_target'] == i, :]
    for v in temp['reg_target'].unique():
        print(i, v, len(temp.loc[temp['reg_target']==v]))

print('#'*50)

### pseudo labeling
sub = pd.read_csv('submission_private.csv')  #predicted results by models for pseudo labeling

plt.hist(sub['classification_predictions'])
plt.show()


# top 20 percent and bottom 20 percent for pseudo labeling
tr0 = np.percentile(sub['classification_predictions'].values, 20)
tr1 = np.percentile(sub['classification_predictions'].values, 80)

pseudo_label0 = sub.loc[(sub['classification_predictions']<=tr0)].copy()
pseudo_label1 = sub.loc[(sub['classification_predictions']>= tr1)].copy()

pseudo_label0['classification_predictions']=0
pseudo_label1['classification_predictions']=1

print(pseudo_label1['regression_predictions'].min(), pseudo_label1['regression_predictions'].shape)
print(pseudo_label0['regression_predictions'].min(), pseudo_label0['regression_predictions'].shape)
plt.hist(pseudo_label1['regression_predictions'])
plt.hist(pseudo_label0['regression_predictions'])
plt.show()

pseudo_label0['regression_predictions'] = np.where(pseudo_label0['regression_predictions'].values>20, 30,
                                                   pseudo_label0['regression_predictions'].values)

pseudo_label0['regression_predictions'] = np.where((pseudo_label0['regression_predictions'].values>9) &
                                                   (pseudo_label0['regression_predictions'].values<12),
                                                   10, pseudo_label0['regression_predictions'].values)

pseudo_label0['regression_predictions'] = np.where((pseudo_label0['regression_predictions'].values<4),
                                                   3, pseudo_label0['regression_predictions'].values)


pseudo_label0 = pseudo_label0.loc[(pseudo_label0['regression_predictions']==30) |
                                  (pseudo_label0['regression_predictions']==10) |
                                  (pseudo_label0['regression_predictions']==3)]

pseudo_label1['regression_predictions'] = np.where((pseudo_label1['regression_predictions'].values>25), 20,
                                                   pseudo_label1['regression_predictions'].values)

pseudo_label1['regression_predictions'] = np.where((pseudo_label1['regression_predictions'].values>5) &
                                                   (pseudo_label1['regression_predictions'].values<9),
                                                   6, pseudo_label1['regression_predictions'].values)

pseudo_label1['regression_predictions'] = np.where(pseudo_label1['regression_predictions'].values<5,
                                                   1, pseudo_label1['regression_predictions'].values)


pseudo_label1 = pseudo_label1.loc[(pseudo_label1['regression_predictions']==20) |
                                  (pseudo_label1['regression_predictions']==6) |
                                  (pseudo_label1['regression_predictions']==1), :]

pseudo_label = pd.concat([pseudo_label0, pseudo_label1], axis=0)

#select integer labels
for lb in [1, 3, 6, 10, 20, 30]:
    for df in [pseudo_label0, pseudo_label1]:
        print('*'*50)
        if df.loc[df['regression_predictions']==lb].shape[0]!=0:
            print(lb, df.loc[df['regression_predictions']==lb].shape[0])

#print pseudo label
pseudo_label.to_csv('pseudo_label.csv', index=False)

print(pseudo_label.shape)
print(pseudo_label.head())

#print pseudo labelled images into training set.
ending = '_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png'

for img_id, cl, reg in tqdm(zip(pseudo_label['id'],
                           pseudo_label['classification_predictions'],
                           pseudo_label['regression_predictions'])):

    name = img_id + f'_{int(reg)}' + ending


    img = cv2.imread(BASE_PATH + 'private_test/' + img_id + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if cl==0:
        save_path = BASE_PATH + 'train/NR/' + name
        cv2.imwrite(filename=save_path, img=img)
    else:
        save_path = BASE_PATH + 'train/ER/' + name
        cv2.imwrite(filename=save_path, img=img)

print('Done')







