import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import os


class ImageDataset(Dataset):

    def __init__(self, train_list, data_path, transform, barcodefile):
        self.train_list = train_list
        self.data_path = data_path
        self.transform = transform
        file_ = open(barcodefile,'r')
        self.barcodes = file_.read().split('\n')[:-1]
        self.num_classes = len(self.barcodes)

    def __len__(self):
        return len(self.train_list)

    def get_num_classes(self):
        return self.num_classes

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.data_path, self.train_list[idx])
        image = Image.open(img_path)
        image_name = self.train_list[idx]
        class_name = image_name.split('_')[0]
        class_id = self.barcodes.index(class_name)
        # classes = []
        # for i in range(self.get_num_classes()):
        #     if i == class_id[0]:
        #         classes.append(1)
        #     else:
        #         classes.append(0)
        image = self.transform(image)
        class_id = torch.tensor(class_id)
        sample = image, class_id
        #print("Classid" ,class_id)
        return sample