"""
Based on https://github.com/sedab/PathCNN
"""

import os
import time
import random
import copy
import numpy as np
import pandas as pd
from PIL import Image
import h5py
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils import new_transforms

class MyTissueData(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, list_IDs, dset_type, transform=None):
        h5_file = h5py.File(hdf5_path)
        if dset_type == 'train':
            self.img_hdf5 = h5_file.get('train_img')
            self.label_hdf5 = h5_file.get('train_labels')
        elif dset_type == 'val':
            self.img_hdf5 = h5_file.get('val_img')
            self.label_hdf5 = h5_file.get('val_labels')
        elif dset_type == 'test':
            self.img_hdf5 = h5_file.get('test_img')
            self.label_hdf5 = h5_file.get('test_labels')
        self.list_IDs = list_IDs
        self.transform=transform

    def __getitem__(self, index):
        idx = self.list_IDs[index]
        img = self.img_hdf5[idx]
        img = Image.fromarray(img)
        label = self.label_hdf5[idx]
        if label ==2:
            label = 1
        elif label ==0:
            label = 0
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.list_IDs)

augment = transforms.Compose([new_transforms.Resize((imgSize, imgSize)),
                              transforms.RandomHorizontalFlip(),
                              new_transforms.RandomRotate(),
                              new_transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def init_model(model):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            if init_method == 'xavier':
                m.weight.data = init.xavier_normal(m.weight.data)
            elif init_method == 'kaiming':
                m.weight.data = init.kaiming_normal(m.weight.data)
            else:
                m.weight.data.normal_(-0.1, 0.1)
            
        elif isinstance(m,nn.BatchNorm2d):
            m.weight.data.normal_(-0.1, 0.1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, pool, **kwargs):
        super(BasicConv2d, self).__init__()

        self.pool = pool
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

        if nonlinearity == 'selu':
            self.relu = nn.SELU()
        elif nonlinearity == 'prelu':
            self.relu = nn.PReLU()
        elif nonlinearity == 'leaky':
            self.relu = nn.LeakyReLU(negative_slope=0.01)
        else:
            self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)

        if self.pool:
            x = F.max_pool2d(x, 2)
        
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class cancer_CNN(nn.Module):
    def __init__(self, nc, imgSize, ngpu):
        super(cancer_CNN, self).__init__()
        self.nc = nc
        self.imgSize = imgSize
        self.ngpu = ngpu
        #self.data = opt.data
        self.conv1 = BasicConv2d(nc, 16, False, kernel_size=5, padding=1, stride=2, bias=True)
        self.conv2 = BasicConv2d(16, 32, False, kernel_size=3, bias=True)
        self.conv3 = BasicConv2d(32, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv4 = BasicConv2d(64, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv5 = BasicConv2d(64, 128, True, kernel_size=3, padding=1, bias=True)
        self.conv6 = BasicConv2d(128, 64, True, kernel_size=3, padding=1, bias=True)
        self.linear = nn.Linear(5184, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    
    return model, best_acc, best_epoch

def test_model(model, loader, dataset_size, criterion):
    
    print('-' * 10)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    whole_probs = torch.FloatTensor(dataset_size)
    whole_labels = torch.LongTensor(dataset_size)
    
    with torch.no_grad():

        for i, data in enumerate(loader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            outputs = F.softmax(outputs, dim=1)
            whole_probs[i*batchSize:i*batchSize+inputs.size(0)]=outputs.detach()[:,1].clone()
            whole_labels[i*batchSize:i*batchSize+inputs.size(0)]=labels.detach().clone()

        total_loss = running_loss / dataset_size
        total_acc = running_corrects.double() / dataset_size

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))

    return whole_probs.cpu().numpy(), whole_labels.cpu().numpy(), total_loss, total_acc

def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUROC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUROC score: [{:0.3f} - {:0.3}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))
    
    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batchSize=32
    init_method="xavier"
    nonlinearity="leaky"
    dropout=0.1
    ngpu=int(1)
    imgSize=int(299)
    nc=int(3)
    num_classes=int(2)
    optim_method="Adam"
    lr=0.001
    beta1=0.5

    hdf5_path = '/path/to/hdf5/with/tumor/annotation/'
    h5 = h5py.File(hdf5_path)

    list_IDs = {}
    for dset_type in ['train', 'val', 'test']:
        if dset_type == 'train':
            list_IDs[dset_type] = [i for i, j in enumerate(h5['train_labels']) if j!=1]
        elif dset_type == 'val':
            list_IDs[dset_type] = [i for i, j in enumerate(h5['val_labels']) if j!=1]
        elif dset_type == 'test':
            list_IDs[dset_type] = [i for i, j in enumerate(h5['test_labels']) if j!=1]

    data = {}
    loaders = {}
    for dset_type in ['train', 'val', 'test']:
        if dset_type == 'train':
            data[dset_type] = MyTissueData(hdf5_path, list_IDs['train'], dset_type='train', transform = augment)
            loaders[dset_type] = torch.utils.data.DataLoader(data[dset_type], batch_size=batchSize, shuffle=True)
        elif dset_type == 'val':
            data[dset_type] = MyTissueData(hdf5_path, list_IDs['val'], dset_type='val', transform = transform)
            loaders[dset_type] = torch.utils.data.DataLoader(data[dset_type], batch_size=batchSize, shuffle=True)
        elif dset_type == 'test':
            data[dset_type] = MyTissueData(hdf5_path, list_IDs['test'], dset_type='test', transform = transform)
            loaders[dset_type] = torch.utils.data.DataLoader(data[dset_type], batch_size=batchSize, shuffle=False)
        print('Finished loading %s dataset: %s samples' % (dset_type, len(data[dset_type])))

    dataset_sizes = {x: len(data[x]) for x in ['train', 'val', 'test']}

    model = cancer_CNN(nc, imgSize, ngpu)
    init_model(model)
    criterion = nn.CrossEntropyLoss()
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    if optim_method == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    elif optim_method == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr = lr)
    elif optim_method == "SGD": 
        optimizer = optim.SGD(model.parameters(), lr = lr)
    else: 
        raise ValueError('Optimizer not found. Accepted "Adam", "SGD" or "RMSprop"') 

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, best_acc, best_epoch = train_model(model, criterion, optimizer, scheduler, num_epochs=25)
    torch.save(model.state_dict(), 'best_checkpoints_epoch_{0}_acc_{1}.pth'.format(str(best_epoch), str(best_acc.item())))

    prob_test, label_test, loss_test, acc_test = test_model(model, loaders['test'], dataset_sizes['test'], criterion)

    bootstrap_auc(label_test, prob_test)

    df = pd.DataFrame(columns=['prob', 'label'])
    df.prob = prob_test
    df.label = label_test
    df.to_csv('/path/to/save/csv', index=False)