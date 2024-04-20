import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader

def acquire_all_patient(dir):
    all_patients = []
    all_group = os.listdir(dir)
    for group in all_group:
        group_path = os.path.join(dir, group)
        group_patients = os.listdir(group_path)
        for patient in group_patients:
            patient_path = os.path.join(group_path, patient)
            all_patients.append(patient_path)
    return all_patients

def acquire_all_img(temp_path,temp_list):

    full_img_path=[]
    full_mask_path=[]

    for patient in temp_list:

        mask_path=os.path.join(temp_path,patient,'mask')
        all_mask=os.listdir(mask_path)
        all_mask=[os.path.join(mask_path,i) for i in all_mask]
        all_img=[i.replace('mask','img') for i in all_mask]
        #all_con=[i.replace('mask','mask_con') for i in all_mask]
        full_img_path.extend(all_img)
        full_mask_path.extend(all_mask)

    return full_img_path,full_mask_path

class BasicDataset(Dataset):
    def __init__(self, img_path,mask_path,scale=1,val=False):
        # self.imgs_dir = imgs_dir
        # self.masks_dir = masks_dir
        self.img_path=img_path
        self.mask_path=mask_path
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.val=val
        self.ids_img=[i for i in range(len(self.img_path))]


    def __len__(self):
        return len(self.ids_img)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, idx):
        # idx = self.ids_img
        mask_file = self.mask_path[idx]
        img_file = self.img_path[idx]


        mask = Image.open(mask_file)
        img = Image.open(img_file)


        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        if not self.val:
            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'name':img_file
            }
        else:
            return  (torch.from_numpy(img).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor),torch.from_numpy(con).type(torch.FloatTensor), self.img_path[idx])



if __name__ == '__main__':
    print('Starting UNet Processing')
    base_dir='../../CAMUS'
    train_path=os.path.join(base_dir,'training')
    test_path=os.path.join(base_dir,'testing')
    train_patints=os.listdir(train_path)
    test_patients=os.listdir(test_path)
    test_img,test_mask=acquire_all_img(test_path,test_patients)
    test_dataset = BasicDataset(test_img,test_mask)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True,drop_last=False)
    i=0
    for batch in test_loader:
        imgs = batch['image']
        mask = batch['mask']
        i += len(imgs)
    print (i)


