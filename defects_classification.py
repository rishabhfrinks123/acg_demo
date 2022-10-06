from __future__ import print_function, division

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm.notebook import tqdm
from PIL import Image



class_names=['broken', 'chipping', 'double', 'empty', 'good', 'spot']

img_dim = (200,130)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_dim),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_dim),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '', 
            result['train_loss'], result['val_loss'], result['val_acc']))

# class TabletsModel(ImageClassificationBase):

class TabletsModel(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = models.efficientnet_b5(pretrained=pretrained)
        
#         print(self.network)
        
#         print(self.network.classifier[1])
        # Replace last layer
        # self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
        self.network.classifier[1]=nn.Linear(self.network.classifier[1].in_features, num_classes)
        
        # print(self.network.fc)

    def forward(self, xb):
        return self.network(xb)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# class DeviceDataLoader():
#     """Wrap a dataloader to move data to a device"""

#     def __init__(self, dl, device):
#         self.dl = dl
#         self.device = device

#     def __iter__(self):
#         """Yield a batch of data after moving it to device"""
#         for b in self.dl:
#             yield to_device(b, self.device)

#     def __len__(self):
#         """Number of batches"""
#         return len(self.dl)


# @torch.no_grad()
# def evaluate(model, val_loader):
#     model.eval()
#     outputs = [model.validation_step(batch) for batch in val_loader]
#     return model.validation_epoch_end(outputs)

# def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
#     history = []
#     optimizer = opt_func(model.parameters(), lr)
#     for epoch in range(epochs):
#         # Training Phase
#         model.train()
#         train_losses = []
#         for batch in tqdm(train_loader):
#             loss = model.training_step(batch)
#             train_losses.append(loss)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#         # Validation phase
#         result = evaluate(model, val_loader)
#         result['train_loss'] = torch.stack(train_losses).mean().item()
#         model.epoch_end(epoch, result)
#         history.append(result)
#     return history

# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']

# def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
#                   weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
#     torch.cuda.empty_cache()
#     history = []

#     # Set up custom optimizer with weight decay
#     optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
#     # Set up one-cycle learning rate scheduler
#     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
#                                                 steps_per_epoch=len(train_loader))

#     for epoch in range(epochs):
#         # Training Phase
#         model.train()
#         train_losses = []
#         lrs = []
#         for batch in tqdm(train_loader):
#             loss = model.training_step(batch)
#             train_losses.append(loss)
#             loss.backward()

#             # Gradient clipping
#             if grad_clip:
#                 nn.utils.clip_grad_value_(model.parameters(), grad_clip)

#             optimizer.step()
#             optimizer.zero_grad()

#             # Record & update learning rate
#             lrs.append(get_lr(optimizer))
#             sched.step()

#         # Validation phase
#         result = evaluate(model, val_loader)
#         result['train_loss'] = torch.stack(train_losses).mean().item()
#         result['lrs'] = lrs
#         model.epoch_end(epoch, result)
#         history.append(result)
#     return history

# device = get_default_device()
# device

# model_complete = TabletsModel(len(class_names))
# to_device(model_complete, device)

def predict_image(im, model,device='cuda'):
    # count=0

    # Convert to a batch of 1
    # im=Image.open(img_path)
    # im= img.resize(img_dim)

    model.eval()
    
    img = data_transforms['val'](im)
    print(img.shape)
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)


    # Pick index with highest probability
    print(yb)
    sm=torch.nn.Softmax()
    prob=sm(yb)
    print(f'probabilityu----{prob}')
    _, preds  = torch.max(yb, dim=1)
    print(preds)
    defect=class_names[preds[0].item()]
    print(f'defect----{class_names[preds[0].item()]}')
    # if class_names[preds[0].item()]=='good':
    #     count+=1
    # print(count)
    return defect
    # plt.imshow(im)



