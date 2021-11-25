import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR


import torchvision
import torchvision.transforms as transforms


from tqdm import tqdm
from model import ViT
from base_data_loader import CustomDataset
from base_trainer import Trainer

import argparse

import math

from pdb import set_trace as st

def label2str(label):
    label_map = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat',
           4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9:'truck'}
    return label_map[label]

class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.NLLLoss()

    def forward(self, input, target):
        ## convert target to one-hot
        # target = F.one_hot(target, num_classes=10) ## (B, 10)
        log_p = F.log_softmax(input, dim=1) ## (B, 10)

        ## calculate cross-entropy (softmax+log+nllloss = cross entropy)
        ce = self.ce(log_p, target) ## nllloss calculate target to onehot

        all_rows = torch.arange(len(input))
        log_pt = log_p[all_rows, target]
        pt = torch.exp(log_pt)

        ## FL(pt) = focal_term * cross_entropy
        focal_term = (1 - pt)**self.gamma

        loss = focal_term * ce

        return loss.mean
    

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="setting using gpu",
                    action="store_true")
parser.add_argument("--test", help="setting using gpu",
                    action="store_true")
parser.add_argument("--gpu", help="setting using gpu",
                    action="store_true")
args = parser.parse_args()
use_gpu = args.gpu
TRAINING = args.train
TESTING = args.test


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

## prepare dataset (baseline)
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_set_size = int(len(train_dataset) * 0.8)
valid_set_size = len(train_dataset) - train_set_size
train_dataset, valid_dataset = data.random_split(train_dataset, [train_set_size, valid_set_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                         shuffle=False, num_workers=0)
## data (N, 3, 224, 224), label (0 - 10)

print('Train dataset: ', len(train_dataset))
print('Test dataset:', len(test_dataset))

# ## prepare dataset (Custom)
# dataset_dir = './'
# custom_dataset = CustomDataset(dataset_dir, transform)
# custom_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=1,
#                                          shuffle=False, num_workers=0)



## define Model
model = ViT(
    image_size=224,
    patch_size=16,
    dim=128,
    num_classes=10,
    channels=3,
    depth = 3,
    heads = 8,
    mlp_dim = 128,
    dropout = 0.1,
    emb_dropout = 0.1
)

## setting Training parameters
save_path = './model.pt'
epochs = 10
lr = 3e-5
gamma = 0.7
# loss function
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

if TRAINING:
    ## training
    trainer = Trainer(model, criterion, optimizer, scheduler, use_gpu)
    trainer.train(epochs, train_loader, valid_loader, save_path)

if TESTING:
    ## inference
    # load model
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

    for data, label in tqdm(custom_loader):
        if use_gpu:
            data = data.cuda()
            label = label.cuda()
        else:
            data = data
            label = label    

        infer_output = model(data)
        pred_label = infer_output.argmax(dim=1)
        print('Ground Truth:', int(label), label2str(int(label)))
        print('Predict:', int(pred_label), label2str(int(pred_label)))