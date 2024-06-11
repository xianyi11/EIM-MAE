import torch.nn.functional
import torch.utils.data as data
from PIL import Image
import os
import torch
from glob import glob
import numpy as np


def get_train_test_dataset(data_root, target_data_id, transform=None):
    train_data_all = []
    train_label_all = []
    test_data_all = []
    test_label_all = []
    target_data_all = []
    target_label_all = []
    data_paths =  glob(data_root+"/*")
    for i,dir in enumerate(data_paths):
        session_dir = glob(dir+"/*")
        for j,dir in enumerate(session_dir):
            id = dir.split("\\")[-1]
            train_data = np.load(os.path.join(dir,"train_data.npy"))
            train_label = np.load(os.path.join(dir,"train_label.npy"))

            test_data = np.load(os.path.join(dir,"test_data.npy"))
            test_label = np.load(os.path.join(dir,"test_label.npy"))
            train_data_all.append(torch.from_numpy(train_data))
            test_data_all.append(torch.from_numpy(test_data))
            train_label_all.append(torch.from_numpy(train_label))
            test_label_all.append(torch.from_numpy(test_label))

    train_data = torch.cat(train_data_all,dim=0)
    train_label = torch.cat(train_label_all,dim=0)
    # target_data = torch.cat(target_data_all,dim=0)
    # target_label = torch.cat(target_label_all,dim=0)
    test_data = torch.cat(test_data_all,dim=0)
    test_label = torch.cat(test_label_all,dim=0)
    train_data = (train_data - train_data.mean())/(train_data.std()*2)
    # target_data = (target_data - train_data.mean())/(train_data.std()*2)
    test_data = (test_data - test_data.mean())/(test_data.std()*2)
    print("train_data",train_data.shape)
    print("train_label",train_label.shape)
    print("test_data",test_data.shape)
    print("test_label",test_label.shape)
    # print("target_data",target_data.shape)
    # print("target_label",target_label.shape)
    train_dataset = GetLoader(train_data, train_label, transform)
    test_dataset = GetLoader(test_data, test_label, transform)
    # target_dataset = GetLoader(target_data, target_label, transform)
    return train_dataset, test_dataset


class GetLoader(data.Dataset):
    def __init__(self, data, label, transform=None):
        B,N,C = data.shape
        self.data = data.float().transpose(1,2)
        self.data = torch.nn.functional.pad(self.data,(0,2)).reshape(B,C,8,8)

        self.label = label.long()

        self.transform = transform

    def __getitem__(self, item):
        imgs = self.data[item]
        labels = self.label[item]

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.data.shape[0]
