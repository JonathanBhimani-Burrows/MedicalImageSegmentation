import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math
import os
import torchvision.transforms as transforms


def dataset_processing(path, names, batch_size):
    '''
    a processing function that creates the datasets and dataloaders with splits
    :param path: path to data
    :param names: names of images in path
    :param batch_size: obvious
    :return: list of dataloaders - train, val then test
    '''
    dl = []
    for idx,p in enumerate(path[1]):
        ds = segdataset(os.path.join(path[0],p),(names[0][idx],names[1][idx]),p)
        if p == 'ct_train':
            splits = split_dataset(len(ds), percentage=0.8)
            for i in splits:
                dl.append(create_dataloader(ds, batch_size, i, sampler=SubsetRandomSampler))
        else:
           dl.append(create_dataloader(ds, batch_size, np.arange(len(ds)), sampler=SubsetRandomSampler))
    return dl



def create_dataloader(dataset, batch_size,idx, sampler=SubsetRandomSampler):
    '''
    returns dataloader
    '''
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,sampler=sampler(idx))


def split_dataset(len_dataset, percentage=0.75):
    '''
    a function that splits the dataset (for train/val split) and shuffles it and returns the splits
    :param len_dataset: length of dataset
    :param percentage: what percent is train vs val
    :return: splits
    '''
    #if shortened dataset
    # a = np.arange(len_dataset)[:200]
    a = np.arange(len_dataset)
    np.random.shuffle(a)
    tr_split = math.floor(len(a) * percentage)
    return (a[:tr_split], a[tr_split:])


class segdataset(Dataset):
    '''
    dataset class for image segmentation
    '''
    def __init__(self, data_path, names, st):
        self.folder = data_path
        self.img_names = names[0]
        self.lbl_names = names[1]
        self.set = st


    def __len__(self):
        '''
        :return: length of dataset
        '''
        return len(self.img_names)

    def __getitem__(self, idx):
        '''
        method that retrieves the image and label for the dataloader
        :param idx: the index used to obtain the image/label
        :return: image and label
        '''
        img_name = os.path.join(self.folder, self.img_names[idx])
        im = transforms.ToTensor()(np.load(img_name))
        if self.set == 'ct_train':
            lbl_name = os.path.join(self.folder, self.lbl_names[idx])
            lbl = np.load(lbl_name)
            if lbl.dtype != 'int16':
                lbl = lbl.astype('int16')
            lbl = transforms.ToTensor()(lbl)
        #check this
        return im, img_name, lbl if self.set == 'ct_train' else np.ones((1,1))


if __name__ == '__main__':
    from data_prep import load_and_save
    s = r"C:\Users\jonat\Documents\Admin\Applications\Circle\data"
    p = ["ct_train"]
    n = load_and_save(s, p, norm='per_patient')
    testing_dataset = segdataset(os.path.join(s,p[0]), (n[0][0],n[1][0]),p[0])
    testing_dataloader = DataLoader(testing_dataset)
    print(len(testing_dataset))
    print(iter(testing_dataloader).next())