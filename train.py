from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from classifier import Classifier
from dataset import ImageDataset
from config import TrainingParameters, DataPaths

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

numworkers = TrainingParameters["num_workers"]
batchsize = TrainingParameters["batch_size"]
PATH = DataPaths["save_model_path"]
data_path = DataPaths["data_path"]
barcodefile = DataPaths["barcode_file_path"]
images_name = os.listdir(data_path)
percentual = (len(images_name)*TrainingParameters["train_val_split"])//100
np.random.shuffle(images_name)
train_data = images_name[percentual:]
val_data = images_name[:percentual]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Scale(256),  # smaller side resized
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}
trainset = ImageDataset(train_data, data_path, data_transforms["train"], barcodefile)
testset = ImageDataset(val_data, data_path, data_transforms["validation"], barcodefile)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=numworkers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=numworkers)
classifier = Classifier(trainset.get_num_classes())
model = classifier.getModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
#model.load(opt.model_save_path)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=TrainingParameters["learning_rate"], momentum=TrainingParameters["momentum"])


for epoch in range(1, 1000+1):
    model.train()
    with torch.set_grad_enabled(True):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(trainloader.dataset),100. * batch_idx / len(trainloader), loss))
    if epoch % 1 == 0:
        model.eval()
        test_loss = 0
        correct = 0
        with torch.set_grad_enabled(False):
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(testloader.dataset)
            print ('Validation set: Average loss:{:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(testloader.dataset), 100.*correct/len(testloader.dataset)))
        torch.save(model.state_dict(), os.path.join(PATH,str(epoch)+'.weights'))
