# Training size 5913
# Accuracy: 83.7 (val) - Resnet101; Fine tuning: Layer4[0,1,2]; fc x 1

# Ref: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from .models import resnet_short, skip_net
import numpy as np
import time
import copy
import PIL
import os


#-----------------
# BASIC INFORMATION
#-----------------

# Use shortened ResNet or Skipnet
use_skipnet = False

# Using pretrained model
use_pretrained_feature_extraction_model = False
use_pretrained_fine_tuning_model = False # Setting it true makes the feature extraction be ignored.

# Dataset dependent variables
range_train = [0,2100]
range_val = [2100,2632]
x_train_path = os.path.expanduser('~/data/PHI/t6/X_train.npy')
y_train_path = os.path.expanduser('~/data/PHI/t6/y_train.npy')
model_fe_filename = 't7_resnet_v2_fe6.ckpt'
model_ft_filename = 't7_resnet_v2_ft6.ckpt'


#-----------------
# HYPER PARAMETERS
#-----------------
# Interation
num_epochs_feature_extraction = 0
num_epochs_fine_tuning = 10
batch_size = 128

# Optimizer
optimizer_type = 'SGD'
initial_lr = 0.02
initial_lr_finetuning = 0.02 # 0.0248364446993
weight_decay = 1e-5 # 1e-2 returns good validation results
momentum = 0.90 #SGD #[0.5, 0.9, 0.95, 0.99]
nesterov = False #SGD

# Learning rate scheduler
scheduler_type = 'StepLR' # ReduceLROnPlateau easily reduces LR even when training error is quite high
step_size = 10 #StepLR
gamma = 0.3  #0.140497184025 #StepLR
factor = 0.3 #ReduceLROnPlateau
patience = 5 #ReduceLROnPlateau
threshold = 5e-2 #ReduceLROnPlateau

# Model
if use_skipnet:
	basemodel = skip_net(pretrained=True)
else:
	basemodel = resnet_short(pretrained=True)

# Transform module
transform_train = transforms.Compose([
                                transforms.RandomHorizontalFlip(), # Hor, Rot >> Make conversion very slow
                                # transforms.RandomVerticalFlip(),   # It seems like adding dataset would be more efficient
                                transforms.RandomApply([
                                        # transforms.RandomRotation(45),
                                        # transforms.RandomAffine(0,translate=(0.1,0.1)),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                                        transforms.RandomGrayscale(),
                                        # transforms.RandomCrop(224,padding=10,pad_if_needed=True),
                                        # transforms.RandomResizedCrop(224,scale=(0.95, 1.0),ratio=(0.95,1.05)),
                                        ], p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform_val = transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



#-----------------
# HELPER FUNCTIONS
#-----------------
# Custom pytorch data loader
class CustomDataset(torch.utils.data.Dataset):
        # Read data from the given path
        def __init__(self, x_path, y_path, isTrain=True, range=None, augmentation=False):
                self.x = np.load(x_path)
                self.y = np.load(y_path)
                self.isTrain = isTrain
                if range != None:
                        self.x = self.x[range[0]:range[1]]
                        self.y = self.y[range[0]:range[1]]
                if augmentation:
                        # Image aug
                        x_90 = np.zeros(self.x.shape, dtype=np.uint8) 
                        # x_180 = np.zeros(self.x.shape, dtype=np.uint8)
                        # x_270 = np.zeros(self.x.shape, dtype=np.uint8) 
                        # x_ver = np.zeros(self.x.shape, dtype=np.uint8)
                        # x_hor = np.zeros(self.x.shape, dtype=np.uint8) 
                        # x_90_hor = np.zeros(self.x.shape, dtype=np.uint8)
                        # x_270_hor = np.zeros(self.x.shape, dtype=np.uint8)
                        for i in np.arange(len(self.x)):
                                img = PIL.Image.fromarray(self.x[i])
                                x_90[i] = np.array(img.transpose(PIL.Image.ROTATE_90))
                                # x_180[i] = np.array(img.transpose(PIL.Image.ROTATE_180))
                                # x_270[i] = np.array(img.transpose(PIL.Image.ROTATE_270))
                                # x_ver[i] = np.array(img.transpose(PIL.Image.FLIP_TOP_BOTTOM))
                                # x_hor[i] = np.array(img.transpose(PIL.Image.FLIP_LEFT_RIGHT))
                                # x_90_hor[i] = np.array((img.transpose(PIL.Image.ROTATE_90)).transpose(PIL.Image.FLIP_LEFT_RIGHT))
                                # x_270_hor[i] = np.array((img.transpose(PIL.Image.ROTATE_270)).transpose(PIL.Image.FLIP_LEFT_RIGHT))
                        self.x = np.concatenate((self.x,x_90), axis=0)
                        self.y = np.concatenate((self.y,self.y), axis=0)
                        permutation = np.random.permutation(self.x.shape[0])
                        self.x = self.x[permutation]
                        self.y = self.y[permutation]

        # Return data (x,y) at the index
        def __getitem__(self, index):
                if self.isTrain:
                        transform = transform_train
                else:
                        transform = transform_val
                x_data = transform(PIL.Image.fromarray(self.x[index]))
                y_data = self.y[index]
                return x_data, y_data
        # Return the length of dataset
        def __len__(self):
                return self.y.shape[0]


#-----------------
# MODEL TRAINING
#-----------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        # Manage the best model so far
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 10000000

        for epoch in range(num_epochs):
                start = time.time()
                print('Epoch {}/{}'.format(epoch+1, num_epochs))
                print('-' * 10)

                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
                ### Training phase
                model.train()  # Set model to training mode
                running_loss = 0.0
                running_corrects = 0
                incorrect = {'00':0, '01':0, '02':0, '03':0, '10':0, '11':0, '12':0, '13':0, '20':0, '21':0, '22':0, '23':0, '30':0, '31':0, '32':0, '33':0}
                # Update learning rate: StepLR
                if type(scheduler) == torch.optim.lr_scheduler.StepLR:
                        scheduler.step()
                        print("Current learning rate: ", scheduler.get_lr())
                for i, (inputs, labels) in enumerate(train_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        with torch.set_grad_enabled(True):
                                outputs = model(inputs)
                                # preds = torch.round(outputs.squeeze()).long()
                                _, preds = torch.max(outputs, 1)
                                loss = criterion(outputs, labels)
                                # backward + optimize only if in training phase
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds.data == labels.data)

                        for j in range(len(labels.data)):
                                key = str(int(labels.data[j])) + str(int(preds.data[j]))
                                incorrect[key] += 1

                # Calculate accuracy
                train_loss = running_loss / len(train_dataset)
                train_acc = running_corrects.double() / len(train_dataset)
                print(incorrect)

                ### Validation phase
                model.eval()   # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                incorrect = {'00':0, '01':0, '02':0, '03':0, '10':0, '11':0, '12':0, '13':0, '20':0, '21':0, '22':0, '23':0, '30':0, '31':0, '32':0, '33':0}
                # Iterate over data.
                for i, (inputs, labels) in enumerate(val_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(False):
                                outputs = model(inputs)
                                # preds = torch.round(outputs.squeeze()).long()
                                _, preds = torch.max(outputs, 1)
                                loss = criterion(outputs, labels)
                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds.data == labels.data)

                        for j in range(len(labels.data)):
                                key = str(int(labels.data[j])) + str(int(preds.data[j]))
                                incorrect[key] += 1

                # Calculate accuracy
                val_loss = running_loss / len(val_dataset)
                val_acc = running_corrects.double() / len(val_dataset)
                print(incorrect)
                print('{} Loss: {:.4f} Acc: {:.4f} || {} Loss: {:.4f} Acc: {:.4f}'.format('train', train_loss, train_acc, 'val', val_loss, val_acc))

                # deep copy the model
                if val_loss < best_loss:
                        print("** New best! **")
                        best_loss = val_loss
                        best_acc = val_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(best_model_wts, 't1_temp_model6_best.ckpt')
                else:
                        print("best so far Loss: {:.4f} Acc: {:.4f}".format(best_loss, best_acc))

                # Update learning rate: ReduceLROnPlateau
                if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                        scheduler.step(train_loss)

                end = time.time() - start
                print('elapsed time: {0} seconds'.format(int(end)))
                print()

                torch.save(model.state_dict(), 't1_temp_model6.ckpt')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


#-----------------
# SETUP MODEL AND DATA
#-----------------
# Dataset
train_dataset = CustomDataset(x_train_path, y_train_path, True, range_train, True)
val_dataset = CustomDataset(x_train_path, y_train_path, False, range_val)

# Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=16, batch_size=batch_size, shuffle=False)

# Enable cuda, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model
model_conv = basemodel
model_conv = model_conv.to(device)


#-----------------
# FEATURE EXTRACTION
#-----------------
if use_pretrained_feature_extraction_model:
        model_conv.load_state_dict(torch.load(model_fe_filename))

ct = 0
for child in model_conv.children():
        print("Child %i:"%ct,child)
        ct += 1
        if ct < 9:
                for param in child.parameters():
                        param.requires_grad = False
        else:
                for param in child.parameters():
                        param.requires_grad = True

criterion = nn.CrossEntropyLoss()

if optimizer_type == 'SGD':
        optimizer_conv = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=initial_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
elif optimizer_type == 'Adam':
        optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=initial_lr, weight_decay=weight_decay)

if scheduler_type == 'StepLR':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)
elif scheduler_type == 'ReduceLROnPlateau':
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv, patience=patience, threshold=threshold, factor=factor, verbose=True)

print()
print("START FEATURE EXTRACTION")
print("************************")
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs_feature_extraction)
torch.save(model_conv.state_dict(), model_fe_filename)


#-----------------
# FINE TUNING
#-----------------
# Fix the CNN params
# ct = 0
# for child in model_conv.children():
#       ct += 1
#       if ct < 8:
#               for param in child.parameters():
#                       param.requires_grad = False
#       else:
#               for param in child.parameters():
#                       param.requires_grad = True

def set_block_training(block=model_conv.layer3)
	for i in range(len(block)):
	        block[i].conv1.weight.requires_grad = True
	        block[i].bn1.weight.requires_grad = True
	        block[i].bn1.bias.requires_grad = True
	        block[i].conv2.weight.requires_grad = True
	        block[i].bn2.weight.requires_grad = True
	        block[i].bn2.bias.requires_grad = True
	        block[i].conv3.weight.requires_grad = True
	        block[i].bn3.weight.requires_grad = True
	        block[i].bn3.bias.requires_grad = True
	        block[i].conv1.weight.requires_grad = True


for i in range(0,5):

        criterion = nn.CrossEntropyLoss()

        if optimizer_type == 'SGD':
                optimizer_conv = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=initial_lr_finetuning, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        elif optimizer_type == 'Adam':
                optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=initial_lr_finetuning, weight_decay=weight_decay)

        if i>0:
        	set_block_training(model_conv.layer3)

        if scheduler_type == 'StepLR':
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'ReduceLROnPlateau':
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv, patience=patience, threshold=threshold, verbose=True, factor=factor)

        print()
        print("START FINE TUNING " + str(int(i)))
        print("********************")
        if use_pretrained_fine_tuning_model:
                model_conv.load_state_dict(torch.load(model_ft_filename))
        model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs_fine_tuning)

        model_ft_filename = 't7_resnet_v2_ft6_{0}.ckpt'.format(int(i))
        torch.save(model_conv.state_dict(), model_ft_filename)
        initial_lr_finetuning = initial_lr_finetuning * 0.1 # 0.0248364446993


