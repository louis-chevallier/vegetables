from utillc import *
import os, glob
import argparse
import time

import torchvision
import torchvision.datasets
from torchvision.datasets import *
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v2

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

print_everything()
EKOX(torch.__version__)

BATCH_SIZE=32
EKO()
def train(gd = "/content/gdrive/MyDrive/data") : 
  root = os.path.join(gd, "Vegetable Images")
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  EKOX(device)
  train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
  ])
  # the validation transforms
  valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
  ])

  ds_train = ImageFolder(os.path.join(root, 'train'),
                transform=train_transform)
  ds_test = ImageFolder(os.path.join(root, 'test'),
                transform=valid_transform)
  ds_valid = ImageFolder(os.path.join(root, 'validation'),
                transform=valid_transform)

  train_loader = torch.utils.data.DataLoader(ds_train, 
          batch_size=BATCH_SIZE, shuffle=True,
          num_workers=4, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(ds_test, 
          batch_size=BATCH_SIZE, shuffle=True,
          num_workers=4, pin_memory=True)
  valid_loadr = torch.utils.data.DataLoader(ds_valid, 
          batch_size=BATCH_SIZE, shuffle=True,
          num_workers=4, pin_memory=True)
  EKO()
  #EKOX(model)

  def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

  classes = list(map(os.path.basename, glob.glob(os.path.join(root, "test", "*"))))
  EKOX(classes)
  nmb_classes = len(classes)
  model = mobilenet_v2(weights="DEFAULT")
  model.fc = nn.Linear(1000, nmb_classes)
  model = mobilenet_v2().to(device)

  # get some random training images
  dataiter = iter(train_loader)
  images, labels = next(dataiter)
  EKOX(labels)

  # show images
  imshow(torchvision.utils.make_grid(images))
  # print labels
  print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))
  

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  EKOX(len(train_loader))  
  for epoch in range(50):  # loop over the dataset multiple times
    EKOX(epoch)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 0:    # print every 200 mini-batches
            EKOT(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0
    correct = 0
    total = 0
    EKO()
    with torch.no_grad():
      for data in test_loader:
        images, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    EKOT(f'Accuracy of the network  test images: {100 * correct // total} %')

    torch.save(model.state_dict(), os.path.join(gd, "vegetables_%03d.cpt" % epoch))
    model.eval()
    x = torch.randn(BATCH_SIZE, 3, 224, 224, requires_grad=True)
    torch_out = model(x)
    torch.onnx.export(model,
                  x,                        
                  os.path.join(gd, "vegetables_%03d.onnx" % epoch),
                  export_params=True, 
                  opset_version=10,
                  do_constant_folding=True, 
                  input_names = ['input'],  
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


