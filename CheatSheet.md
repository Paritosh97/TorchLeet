# Pytorch Vision CheatSheet

## Imports

### Torch + Torchvision

```python
# generics
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

# backbones / pretrained:
import torchvision.models

# segmentation models
import segmentation_models_pytorch as smp

# warmup lib
import pytorch_warmup as warmup
```

### Musts + Image Utils

```python
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import numpy as np
import pandas as pd

# augmentation 
import albumentations as A
```

### Extra

```python
# debugger
import ipdb
```

## Reproducibility

```python
# ensure reproducibility
random.seed(23)
torch.manual_seed(23)
np.random.seed(23)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Constants

### ImageNet Normalization

```python
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMEAN = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
ISTD = [1/0.229, 1/0.224, 1/0.225]
```

### Device

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Train Hyper Params

```python
lr = .001
weight_decay = .0001
epochs = 100
```

## Plots

```python
def plot_ss(right, left):
    """ Plot side-by-side """
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(right)
    ax[1].imshow(left)

def plot_samples(samples, cols=4, figsize=(10, 10)):
    """ Plot n samples distributed in given cols and figsize """
    f, ax = plt.subplots(round(n/cols), cols)
    ax = ax.ravel()
    for i, s in enumerate(samples):
        ax[i].imshow(s)
```

## Data Utils

### Image functions

```python
def to_tensor(im):
    """ Pillow image to Tensor """
    transform = T.Compose([
        T.CenterCrop(CROP_SIZE),
        T.Resize(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])
    tensor = transform(im)
    return tensor

def from_tensor(tensor, size):
    """ Pillow image from tensor """
    transform = T.Compose([
        T.ToPILImage(tensor),
        T.Resize(size),
        T.Normalize(IMEAN, ISTD),
    ])

def scaled(image, factor):
    size = int(image.width * factor), int(image.height * factor))
    return image.resize(size)
```

### DataSet Example

```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, input_size, aug=False):
        # prepare images / masks segs
        self.ims = glob.glob(os.path.join(dataset_path, 'images', '*'))
        self.segs = glob.glob(os.path.join(dataset_path, 'masks', '*'))
        assert len(self.ims_paths) == len(self.segs_paths), 'Invalid dataset'

        self.input_size = (input_size, input_size)
        self.aug = aug

    def load_seg(self, segpath):
        ... read & preprocess segmentation mask ...
        return mask

    def load_im(self, impath):
        ... read & preprocess image ...
        return im

    def augment(self, im, seg):
        ... apply augmentation ...
        return im, seg

    def __len__(self):
        return len(self.ims_paths)

    def __getitem__(self, idx):
        # load
        im, seg = self.ims[idx], self.segs[idx]
        if self.aug:
            im, seg = self.augment(im, seg)
        return im, seg
```

### Split dataset into DataLoaders

```python
def split_train_val(dataset, batch_size=64, num_workers=0, split=0.7):
    # create lengths
    size = len(dataset)
    train_size = int(np.ceil(size * split))
    test_size = size - train_size
    lengths = [train_size, test_size]

    # create subsets
    train_set, validation_set = torch.utils.data.random_split(dataset, lengths)

    # create loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    return train_loader, val_loader
```

## Algorithms

### Train

```python
""" train epoch """

def train_epoch(e, model, data_loader, criterion, optimizer, device, writer):
    # prepare
    model.train()
    model.to(device)
    loss, acc = .0, .0
    ... extra metrics initialization ...

    for x, y in data_loader:
        # pass tensor through the net
        x, y = x.to(device), y.to(device)
        out = model(x)

        # loss
        batch_loss = criterion(out, y)
        loss += batch_loss.item() * y.size(0)

        # accuracy
        batch_acc = accuracy(out, y)
        acc += batch_acc * y.size(0)

        # optimize
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        ... compute extra metrics ... 

        # write to tensorboard
        writer.add_scalar('train_loss', batch_loss, e)
        writer.add_scalar('train_acc', batch_acc, e)

    data_len = len(data_loader)
    return loss/data_len, acc/data_len
```

### Validation

```python
""" validation epoch algorithm """

def val_epoch(e, model, data_loader, criterion, device, writer):
    # prepare
    model.eval()
    model.to(device)
    loss, acc = 0., 0.
    ... extra metrics initialization ...

    with torch.no_grad():
        for x, y in data_loader:
            # pass tensor though the net
            x, y = x.to(device).float(), y.to(device).long()
            out = model(x)

            # loss
            batch_loss = criterion(out, y.squeeze(1))
            loss += batch_loss.item() * y.size(0)

            # accuracy
            batch_acc = accuracy(out, y)
            acc += batch_acc * y.size(0)

            ... compute extra metrics ... 

            # write to tensorboard
            writer.add_scalar('val_loss', batch_loss, e)
            writer.add_scalar('val_acc', batch_acc, e)

    data_len = len(data_loader.dataset)
    return loss/data_len, acc/data_len, ... extra ...
```

### Run epochs & plot history

```python
# start the tensorboard logs
writer = SummaryWriter('logs/', flush_secs=5)
%tensorboard --logdir logs/ --reload_interval 5

# init history
history = {
    'train': {'loss': [], 'acc': []},
    'val': {'loss': [], 'acc': []}
}

# train!
for e in range(epochs):
    # run epoch
    train_metrics = train_epoch(e, unet, train_loader, criterion,
                              optimizer, device, writer)
    val_metrics = val_epoch(e, unet, val_loader, criterion, device, writer)

    train_loss, train_iou = train_metrics
    val_loss, val_iou = val_metrics

    # history logging
    history['train']['loss'].append(train_loss)
    history['train']['acc'].append(train_iou)
    history['val']['loss'].append(val_loss)
    history['val']['acc'].append(val_iou)

writer.close()

# plot history
f, ax = plt.subplots(1, 2, figsize=(15, 5))
for i, name in enumerate(['loss', 'acc']):
    ax[i].set_title(name)
    ax[i].plot(history['train'][name], label=f'train {name}')
    ax[i].plot(history['val'][name], label=f'val {name}')
    ax[i].legend()
plt.show()

# show final results
final_loss = history['val']['loss'][-1]
final_iou = history['val']['iou'][-1]
print(f'Eval loss: {final_loss} - Eval iou: {final_iou}')
```