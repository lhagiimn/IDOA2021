from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import os
import pathlib as path

def img_loader(path: str):
    with Image.open(path) as img:
        img = img.convert('RGB')
    return img


#training dataset
class IDOADataset(Dataset):
    def __init__(self, imgs, base_path, transforms, train=True):

        self.imgs = imgs
        self.base_path = base_path
        self.transforms = transforms
        self.train = train

    def __getitem__(self, index):

        im_path = self.base_path+'ER/'+self.imgs[index]
        if os.path.isfile(im_path):
            img = img_loader(im_path)
            class_target = torch.tensor(1.0)
        else:
            im_path = self.base_path+'NR/'+self.imgs[index]
            img = img_loader(im_path)
            class_target = torch.tensor(0.0)

        if self.transforms:
            img = self.transforms(img)

        if self.train:
            names = os.path.split(self.imgs[index])[-1].split("_")
            idx = [i for i, v in enumerate(names) if v == "keV"][0]
            reg_target = torch.tensor(float(names[idx - 1]))

            return img, (reg_target, class_target)
        else:
            return img

    def __len__(self):
        return len(self.imgs)

#image augmentation
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(brightness=(0.2, 2),
    #                        contrast=(0.3, 2),
    #                        saturation=(0.2, 2),
    #                        hue=(-0.3, 0.3)),
    transforms.RandomAffine(30),
    transforms.RandomGrayscale(p=1),
    #transforms.RandomPerspective(),
    transforms.RandomRotation(90, expand=False),
    transforms.CenterCrop((64, 64)), #for pseudo labelling model it should be equal to (96, 96)
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.CenterCrop((64, 64)), #for pseudo labelling model it should be equal to (96, 96)
    transforms.ToTensor(),
])


#test dataset
class IDOATestDataset(Dataset):
    def __init__(self, base_path, transforms, private=True):

        self.base_path = base_path
        self.transforms = transforms
        self.len = len(list(path.Path(self.base_path + 'private_test').glob("*.png")))
        if private==True:
            self.imgs= list(path.Path(self.base_path + 'private_test').glob("*.png"))
        else:
            self.imgs = list(path.Path(self.base_path+'private_test').glob("*.png"))+\
                        list(path.Path(self.base_path + 'public_test').glob("*.png"))

    def __getitem__(self, index):

        id = self.imgs[index]
        img = img_loader(id)
        img = self.transforms(img)
        len = self.len

        return img, id.name.split(".")[0], len

    def __len__(self):
        return len(self.imgs)

