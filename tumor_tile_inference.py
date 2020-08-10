import os
import numpy as np
import pandas as pd
from PIL import Image
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils import new_transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, transform=None):
        h5_file = h5py.File(hdf5_path)
        self.imgs = h5_file.get('img')
        self.fnames = h5_file.get('fnames')
        self.ids = h5_file.get('ids')
        self.slides = h5_file.get('slides')
        self.events = h5_file.get('pfi')
        self.times = h5_file.get('pfitime')
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index])
        idx = self.ids[index].decode('UTF-8')
        slide = self.slides[index].decode('UTF-8')
        fname = self.fnames[index].decode('UTF-8')
        event = self.events[index]
        time = self.times[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, idx, slide, fname, event, time

    def __len__(self):
        return len(self.ids)

transform = transforms.Compose([new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

def test_model(model, loader, dataset_size):
    
    model.eval()
    
    wprobs = torch.FloatTensor(dataset_size)
    wpreds = torch.LongTensor(dataset_size)
    idxs = []
    slides = []
    fnames = []
    events = []
    times = []
    
    with torch.no_grad():
        for i, data in enumerate(loader):

            inputs = data[0].to(device)
            idx = data[1]
            slide = data[2]
            fname = data[3]
            event = data[4]
            time = data[5]

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            outputs = F.softmax(outputs, dim=1)
            
            wprobs[i*batchSize:i*batchSize+inputs.size(0)]=outputs.detach()[:,1].clone()
            wpreds[i*batchSize:i*batchSize+inputs.size(0)]=preds.detach().clone()
            
            for j in range(inputs.size(0)):
                idxs.append(idx[j])
                slides.append(slide[j])
                fnames.append(fname[j])
                events.append(event[j])
                times.append(time[j])
            
    return wprobs.cpu().numpy(), wpreds.cpu().numpy(), idxs, slides, fnames, events, times

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    np.random.seed(123456)
    _ = torch.manual_seed(123456)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batchSize=32
    nonlinearity="leaky"
    dropout=0.1
    ngpu=int(1)
    imgSize=int(299)
    nc=int(3)
    num_classes=int(2)

    hdf5_path = '/path/to/tumor_tile_inference.hdf5/'

    dset = MyDataset(hdf5_path, transform)
    dloader = torch.utils.data.DataLoader(dset, batch_size=batchSize, shuffle=False)
    print('Finished loading dataset: %s samples' % len(dset))
    dset_size = len(dset)

    model = cancer_CNN(nc, imgSize, 1)
    model.cuda()
    model_path = '/path/to/checkpoint/'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    probs, preds, idxs, slides, fnames, events, times = test_model(model, dloader, dset_size)

    df = pd.DataFrame(columns=['id', 'slide', 'fname', 'event', 'time', 'prob', 'pred'])
    events = [int(i.numpy()) for i in events]
    times = [float(i.numpy()) for i in times]
    df.id = idxs
    df.slide = slides
    df.fname = fnames
    df.event = events
    df.time = times
    df.prob = probs
    df.pred = preds
    df.to_csv('/path/to/save/tumor_tile_inference_result.csv', index=False)