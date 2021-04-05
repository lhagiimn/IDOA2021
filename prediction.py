import pandas as pd
from Dataset import *
from tqdm import tqdm
import numpy as np

def postprocess(df, class_label):

    if class_label==0:
        tr0 = 20
        max_tr1, min_tr1 = 15, 7
        max_tr2 = 6
        df['regression_predictions'] = np.where(df['regression_predictions'].values>tr0, 30, df['regression_predictions'].values)
        df['regression_predictions'] = np.where((df['regression_predictions'].values>min_tr1) & (df['regression_predictions'].values<max_tr1), 10, df['regression_predictions'].values)
        df['regression_predictions'] = np.where(df['regression_predictions'].values<max_tr2, 3, df['regression_predictions'].values)
    elif class_label==1:
        tr0 = 17
        max_tr1, min_tr1 = 12, 4
        max_tr2 = 3
        df['regression_predictions'] = np.where(df['regression_predictions'].values > tr0, 20, df['regression_predictions'].values)
        df['regression_predictions'] = np.where((df['regression_predictions'].values > min_tr1) & (df['regression_predictions'].values < max_tr1), 6,df['regression_predictions'].values)
        df['regression_predictions'] = np.where(df['regression_predictions'].values < max_tr2, 1, df['regression_predictions'].values)
    else:
        tr0 = 27
        max_tr1, min_tr1 = 23, 16
        max_tr2, min_tr2 = 13, 8
        max_tr3, min_tr3 = 7, 5
        max_tr4, min_tr4 = 4, 2.5
        max_tr5 = 1.75

        df['regression_predictions'] = np.where(df['regression_predictions'].values > tr0, 30,df['regression_predictions'].values)
        df['regression_predictions'] = np.where((df['regression_predictions'].values > min_tr1) & (df['regression_predictions'].values < max_tr1), 20, df['regression_predictions'].values)
        df['regression_predictions'] = np.where((df['regression_predictions'].values > min_tr2) & (df['regression_predictions'].values < max_tr2), 10, df['regression_predictions'].values)
        df['regression_predictions'] = np.where((df['regression_predictions'].values > min_tr3) & (df['regression_predictions'].values < max_tr3), 6, df['regression_predictions'].values)
        df['regression_predictions'] = np.where((df['regression_predictions'].values > min_tr4) & (df['regression_predictions'].values < max_tr4), 3, df['regression_predictions'].values)
        df['regression_predictions'] = np.where(df['regression_predictions'].values < max_tr5, 1, df['regression_predictions'].values)

    return df


## predicting and print submission csv file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = 'idao_dataset/track_1_train_public/' #track 2 BASE_PATH should be tests

Pseudo_labeling = False     # If True it will perform a pseudo labelling

if Pseudo_labeling==True:
    Only_private_set = True  # If True only private set will be predicted.
else:
    Only_private_set = False  # If True only private set will be predicted. We can change here.

if Pseudo_labeling==True:
    MODEL_PATH = 'models'  #the path of models for preudo labelling
    fold_range = 2
else:
    MODEL_PATH = 'final_models' #the path of models for final prediciton
    fold_range = 3

test_set = IDOATestDataset(base_path=BASE_PATH, transforms=test_transform, private=Only_private_set)
test_loader= DataLoader(dataset=test_set, batch_size=32, shuffle=False)

# pseudo model in models folder and final model in final_models folder

models_reg = []
models_cl = []
for fold in range(1, fold_range):
    model = torch.load(f'{MODEL_PATH}/model_regression_{fold}.pth',  map_location=device)
    models_reg.append(model)
    model = torch.load(f'{MODEL_PATH}/model_classification_{fold}.pth', map_location=device)
    models_cl.append(model)


test_preds_class = []
test_preds_reg = []
idx = []

for img, id, pl in tqdm(test_loader):

    idx += list(id)
    temp_reg_pred = np.zeros(img.shape[0])
    temp_cl_pred = np.zeros(img.shape[0])
    for i, (model_reg) in enumerate(models_reg):
        model_reg.eval()
        with torch.no_grad():
            reg_out = model_reg(img.to(device))
            temp_reg_pred += reg_out.detach().cpu().numpy()[:, 0] / len(models_reg)

    test_preds_reg += list(temp_reg_pred)

    for i, (model_cl) in enumerate(models_cl):
        model_cl.eval()
        with torch.no_grad():
            class_out = model_cl(img.to(device))
            temp_cl_pred +=class_out.sigmoid().detach().cpu().numpy()[:, 0]/len(models_cl)

    test_preds_class += list(temp_cl_pred)

length = pl.detach().cpu().numpy()[0]

df = pd.DataFrame()
df['id'] = idx
df['classification_predictions'] = test_preds_class
df['regression_predictions'] = test_preds_reg


#post processing

if Only_private_set==False:
    public_df = df.iloc[length:].copy()
    private_df = df.iloc[:length].copy()

    tr = np.percentile(df['classification_predictions'].values, 50)

    class_label0 = private_df.loc[(private_df['classification_predictions'] <= tr)].copy()
    class_label1 = private_df.loc[(private_df['classification_predictions'] > tr)].copy()

    class_label0 = postprocess(class_label0, class_label=0)
    class_label1 = postprocess(class_label1, class_label=1)

    private_df = pd.concat([class_label0, class_label1], axis=0)

    public_df = postprocess(public_df, class_label=2)
    df = pd.concat([private_df, public_df], axis=0)
    df.to_csv('submission.csv', index=False)

else:

    df.to_csv('submission_private.csv', index=False)





