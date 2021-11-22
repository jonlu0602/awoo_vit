import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR


import torchvision
import torchvision.transforms as transforms



from tqdm import tqdm
from model import ViT

from pdb import set_trace as st

def label2str(label):
    label_map = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat',
           4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9:'truck'}
    return label_map[label]

## prepare dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 8
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Random split
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


## define Model
model = ViT(
    image_size=224,
    patch_size=32,
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
epochs = 1
lr = 3e-5
gamma = 0.7
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


# for epoch in range(epochs):
#     epoch_loss = 0
#     epoch_accuracy = 0

#     for data, label in tqdm(train_loader):
#         data = data
#         label = label

#         output = model(data)
#         loss = criterion(output, label)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         acc = (output.argmax(dim=1) == label).float().mean()
#         epoch_accuracy += acc / len(train_loader)
#         epoch_loss += loss / len(train_loader)
        

#     with torch.no_grad():
#         epoch_val_accuracy = 0
#         epoch_val_loss = 0
#         for data, label in valid_loader:
#             data = data
#             label = label

#             val_output = model(data)
#             val_loss = criterion(val_output, label)

#             acc = (val_output.argmax(dim=1) == label).float().mean()
#             epoch_val_accuracy += acc / len(valid_loader)
#             epoch_val_loss += val_loss / len(valid_loader)
#     print(
#         f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
#     )
#     break

## inference

for data, label in tqdm(test_loader):
    data = data
    label = label

    infer_output = model(data)
    pred_label = infer_output.argmax(dim=1)
    print('Ground Truth:', label2str(int(label)))
    print('Predict:', label2str(int(pred_label)))
    st()
    # acc = (infer_output.argmax(dim=1) == label).float().mean()
    # print()