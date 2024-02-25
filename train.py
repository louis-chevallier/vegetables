from utillc import *
import os, glob, sys
import argparse
import time
import torchvision
import torchvision.datasets
from torchvision.datasets import *
import torch
import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights, maxvit_t
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import efficientnet_v2_m
from torchvision.models import mobilenet_v2, resnet50
from torchvision.models import vgg16_bn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

print_everything()
EKOX(torch.__version__)

BATCH_SIZE=32
EKO()

SIZE = (224, 224)
SIZE2 = (340, 340)

#SIZE = (340, 340)
#SIZE2 = (448, 448)

dataset = "Vegetable Images"
dataset = "FruitsVegetables"

class Vegetable :
  def mobilenetf(nmb_classes) : 
    model = mobilenet_v2(weights="DEFAULT")

    model.classifier = nn.Sequential(
      #nn.Dropout(p=0.2),
      nn.Linear(1280, 512),
      nn.ReLU(), 
      nn.Linear(512, nmb_classes))
    
    #model.classifier[1] = nn.Linear(1280, nmb_classes)
    EKOX(model)
    return model, model.classifier

  def resnet101f(nmb_classes) :
    ### resnet
    model = resnet101(weights="DEFAULT")
    model.fc = nn.Sequential(
      #nn.Dropout(p=0.2),
      nn.Linear(2048, 512),
      nn.ReLU(), 
      nn.Linear(512, nmb_classes))
    
    return model, model.fc

  def resnetf(nmb_classes) :
    ### resnet
    model = resnet50(weights="DEFAULT")
    #EKOX(model)
    model.fc = nn.Sequential(
      #nn.Dropout(p=0.2),
      nn.Linear(2048, 512),
      nn.ReLU(), 
      nn.Linear(512, nmb_classes))
    return model, model.fc

  def efficientnetf(nmb_classes) :
    model = efficientnet_v2_m(weights="DEFAULT")
    EKOX(model)
    model.classifier[1] = nn.Linear(1280, nmb_classes)
    
    return model, model.classifier

  def maxvitf(nmb_classes) :
    model = maxvit_t(weights="DEFAULT")
    #EKOX(model)
    model.classifier[5] = nn.Linear(512, nmb_classes)
    return model, model.classifier

  def vgg16_bnf(nmb_classes) :
    model = vgg16_bn(weights="DEFAULT")
    EKOX(model)
    model.classifier[6] = nn.Linear(4096, nmb_classes)
    EKOX(model)    
    return model, model.classifier


  t = {
    #"vgg16_bn" : (vgg16_bnf, 32), 
    #"resnet101" : (resnet101f, 32), 
    #"maxvit" : (maxvitf, 32),
    
    #"efficientnet_v2" : (efficientnetf, 32),
    "resnet50" : (resnetf, 64),
    "mobilenet_v2" : (mobilenetf, 64),
  }

  train_transform = transforms.Compose([
    transforms.Resize(SIZE2),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70),
                              interpolation = transforms.InterpolationMode.BILINEAR ),
    transforms.RandomResizedCrop(SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])
  ])

  valid_transform = transforms.Compose([
    transforms.Resize(SIZE2),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.),      
    transforms.RandomResizedCrop(SIZE),      
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])
  ])

  valid_transform = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])
  ])

  inference_transform  = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])
  ])
  


  def __init__(self, gd = "/content/gdrive/MyDrive/data", use_gpu=True,
               model_name="mobilenet_v2",
               train_dir = None) :
    self.gd = gd
    self.train_dir = self.gd if train_dir is None else train_dir
    self.use_gpu = use_gpu
    self.model_name = model_name
    EKO()
    self.load(model_name, disp=False)
    pass

  def get_model(self) :
    return mobilenet_v2 if self.model_name == "mobilenet_v2" else resnet50

  
  def load(self, model_name=None, disp=False) :
    gd = self.gd
    root = os.path.join(self.train_dir, dataset)
    EKOX(torch.cuda.is_available())
    EKOX(self.use_gpu)
    self.device = device = 'cpu' if not self.use_gpu else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    EKOX(device)

    
    t = self.t
    _, batch_size = t[self.model_name]


    ds_train = ImageFolder(os.path.join(root, 'train'),
                           transform=self.train_transform)
    ds_test = ImageFolder(os.path.join(root, 'test'),
                          transform=self.valid_transform)
    ds_valid = ImageFolder(os.path.join(root, 'validation'),
                           transform=self.valid_transform)
    ds_valid2 = ImageFolder(os.path.join(root, 'validation2'),
                            transform=self.valid_transform)
    self.class_to_idx = ds_train.class_to_idx
    self.idx_to_class = dict([ (v,k) for k,v in self.class_to_idx.items()])

    classes = list(self.class_to_idx.keys())
    EKOX(classes)
    nmb_classes = len(classes)
    EKOX(nmb_classes)
    self.classes = classes
    
    model, classifier = t[self.model_name][0](nmb_classes)
    model = model.to(device)
    EKOX(self.model_name)
    total_params = sum(param.numel() for param in model.parameters())
    EKOX(total_params)    
    total_params_train = sum(param.numel() for param in classifier.parameters())
    EKOX(total_params_train)    

    
    EKO()
    #EKOX(model)

    def imshow(img):
      img = img / 2 + 0.5     # unnormalize
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))
      plt.show()

    EKOT("building loader ..")
    train_loader_visu = torch.utils.data.DataLoader(ds_train, 
            batch_size=batch_size*4, shuffle=True,
            num_workers=4, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(ds_train, 
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(ds_test, 
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    valid_loadr = torch.utils.data.DataLoader(ds_valid, 
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    valid_loadr2 = torch.utils.data.DataLoader(ds_valid2, 
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)


    # get some random training images
    dataiter = iter(train_loader_visu)
    images, labels = next(dataiter)
    EKO()
    #EKOX(str(labels))
    if disp :
      # show images
      EKO()
      print(' '.join(f'{self.idx_to_class[int(labels[j])]:5s}' for j in range(BATCH_SIZE)))      
      imshow(torchvision.utils.make_grid(images))

    # print labels

    #EKOX(torchscan.summary(model), (3, SIZE, SIZE))

    
    return model, train_loader, test_loader, valid_loadr, valid_loadr2, batch_size, classifier

  def read(self, epoch, model) :
    path_to_read = os.path.join(self.gd, "models", "vegetables_%s_%s_%03d.cpt" % (dataset, self.model_name, epoch))
    EKOT("using ", path_to_read)
    state = torch.load(path_to_read, map_location=self.device)
    model.load_state_dict(state)
    #EKOX(model)

  def get_classes(self) :
    return list(self.class_to_idx.keys())
    
  
  def test(self, epoch=193, measure=True, disp=False, test_dir=None) :
    """
    measure : faire le calcul de l'accuracy
    """
    EKO()
    model, train_loader, test_loader, valid_loader, valid_loader2, _, classifier = self.load(disp=disp)

    
    EKO()
    self.read(epoch, model)
    EKO()
    if test_dir is not None :

      images = glob.glob("%s/*.jpg" % test_dir)
      #EKOX(images)
      for i in images :
        lab, _, p = self.predict(model, Image.open(i))
        EKOX((i, lab, self.classes[lab], p))
      
    if measure :
      EKOT("train"); self.measure(train_loader, model)
      EKOT("test"); self.measure(test_loader, model)
      EKOT("valid"); self.measure(valid_loader, model)
      EKO("valid2"); self.measure(valid_loader2, model)      
    return model
  
  def measure(self, loader, model) :
    #EKOT(" testing %s" % loader)
    n = len(self.classes)
    confusion_table = np.zeros((n, n)).astype(int)
    device = self.device
    correct = 0
    total = 0
    #EKOT(len(loader))
    with torch.no_grad():
      for data in loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        #EKOX(outputs.data.shape)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i, pred in enumerate(outputs.data) :
          true_label = labels[i].cpu().numpy()
          #EKOX(true_label)
          #EKOX(pred)
          #EKOX(int(torch.max(pred, 0).indices))
          confusion_table[true_label, int(torch.max(pred, 0).indices)] += 1
          
    classes = list(self.class_to_idx.keys())
    classesn = list(map(str, range(len(classes))))
    #EKOX(list(zip(classesn, classes)))
    #EKOX(classesn)
    fs = lambda x : "%3s" % x
    #EKOX("\n   " + '\t'.join(map(fs, classesn)) + '\n' + '\n'.join([ fs(classesn[i]) + "" + "\t".join(map(fs, e)) for i, e in enumerate(confusion_table)]))
    accuracy = 100 * correct // total
    EKOX(accuracy)
    return accuracy
    
  def predict(self, model, image_to_predict) :
    #EKOX(image_to_predict)

    i = self.inference_transform(image_to_predict)[None, ...]
    #EKOX(i.shape)
    images = i.to(self.device)
    #EKOX(model)
    model.eval()
    with torch.no_grad():    
      outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    #EKOX(outputs.shape)
    #EKOX(outputs)
    #EKOX(torch.nn.functional.softmax(outputs, dim=1))
    
    #EKOX(predicted)
    label = predicted.to('cpu').numpy()[0]
    #EKOX(label)
    prob = torch.nn.functional.softmax(outputs, dim=1)[0, label]
    #EKOX(outputs.data.shape)
    #EKOX(outputs.data[0, label])
    #EKOX(label)
    #EKOX(self.idx_to_class[label])
    return label, outputs.data.cpu().numpy(), prob
    
  def train(self) :
    gd = self.gd
    modelg, train_loader, test_loader, valid_loader, valid_loader2, batch_size, classifier = self.load(model_name=self.model_name, disp=False)
    device = self.device
    criterion = nn.CrossEntropyLoss()
    accs, accs_train = [], []
    
    def phase(submodel, lr, start_epoch, end_epoch) :
      """
      on entraine la partie terminale du reseau puis tout le reseau
      """
      optimizer = optim.Adam(submodel.parameters(), lr=lr)
      #optimizer = optim.NAdam(model.parameters(), lr=0.02)
      EKOX(len(train_loader))

      with open("accuracies.txt", "w") as fd :
        fd.write("\n".join(map(str, accs)))

      for epoch in range(start_epoch, end_epoch):  # loop over the dataset multiple times
        EKOX(epoch)
        running_loss = 0.0
        modelg.train()
        modelgg = modelg.to(self.device)
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = modelgg(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 0:    # print every 200 mini-batches
                EKOT(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        accs_train.append(self.measure(train_loader, modelgg))
        accs.append(self.measure(test_loader, modelgg))
        self.measure(valid_loader2, modelgg)
        self.measure(valid_loader, modelgg)


        os.makedirs( os.path.join(gd, "models"), exist_ok=True)
        torch.save(modelgg.state_dict(), os.path.join(gd, "models", "vegetables_%s_%s_%03d.cpt" % (dataset, self.model_name, epoch)))
        modelggg = modelgg.to('cpu')
        modelggg.eval()
        x = torch.randn(BATCH_SIZE, 3, 224, 224, requires_grad=True)
        torch_out = modelggg(x)
        torch.onnx.export(modelggg,
                          x,                        
                          os.path.join(gd, "models", "vegetables_%s_%s_%03d.onnx" % (dataset, self.model_name, epoch)),
                          export_params=True, 
                          opset_version=10,
                          do_constant_folding=True, 
                          input_names = ['input'],  
                          output_names = ['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})


    phase(classifier, 0.0001, 0, 200)
    EKO()
    phase(modelg, 0.00001, 200, 300) 
    EKO()
        
    with open("accuracies_%s.txt" % self.model_name, "w") as fd :
      fd.write("\n".join(map(str, accs)))
    with open("accuracies_train_%s.txt" % self.model_name, "w") as fd :
      fd.write("\n".join(map(str, accs_train)))

def train(gd = "/content/gdrive/MyDrive/data", train_dir=None) :
  models_name = Vegetable.t.keys()
  for mn in models_name :
    v = Vegetable(gd, model_name=mn, train_dir=train_dir)
  EKO()  
  for mn in models_name :
    v = Vegetable(gd, model_name=mn, train_dir=train_dir)
    v.train()


def test(gd = "/content/gdrive/MyDrive/data", train_dir=None, test_dir=None, model_name ="resnet50", epoch=193) :
  #v = Vegetable(gd, use_gpu=True, train_dir=train_dir)
  #v.test(46)
  v = Vegetable(gd, model_name=model_name, train_dir=train_dir, use_gpu=True)
  v.test(measure=True, test_dir = test_dir, epoch=epoch)

def predict(gd = "/content/gdrive/MyDrive/data") :
  v = Vegetable(gd, use_gpu=True)
  model = v.test(measure=False, disp=False)
  model.eval()
  v.predict(model, Image.open('brocoli.jpg'))
  v.predict(model, Image.open('concombre.jpg'))

  l = glob.glob("tests/*.jpg")  
  for f in l :
    im = Image.open(f)
    label, _, prob = v.predict(model, im)
    EKOI(np.asarray(im))
    EKOX(v.idx_to_class[label])
    EKOX(prob)
