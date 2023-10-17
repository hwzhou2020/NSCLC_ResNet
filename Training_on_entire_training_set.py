"""
For each train-test experiment, train a model based on predetermined parameters and test on the reserved slides

@author: Haowen Zhou and Siyu (Steven) Lin, Oct 17, 2023
"""

# In[0] Dependencies and Library Configurations
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import copy
from PIL import Image

from sklearn.metrics import roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')

cudnn.benchmark = True
plt.ion()   # interactive mode


# In[1] Utility Functions
def soft(x,y):
    ''' Softmax of x to [x,y]'''
    return 1/(1 + np.exp(y-x))


def shuffle(slides_BM, slides_C, nfold = 1, num_val = 20):
    ''' Given the Met+ and Met- slides, reserve a balanced set of Met+ and Met- slides for validation/testing 
    
    Parameters:
    slides_BM (int array): Array of the slide numbers with class label Met+(BM)
    slides_C  (int array): Array of the slide numbers with class label Met-(C)
    nfold (int)          : Number of folds
    num_val (int)        : Total number of balanced Met+ and Met- slides

    Returns:
    int array with size (nfold, num_val) : Returning the slide numbers of the reserved balanced slides for each fold
    '''

    validation_slide_split = np.zeros((nfold, num_val))
    np.random.seed(1)        # Random Seed, allows for replicating the result
    np.random.shuffle(slides_BM)
    np.random.shuffle(slides_C)
    for fold in range(nfold):
        validation_slide_split[fold][:] = np.concatenate((np.array(slides_BM[fold * (num_val // 2): fold * (num_val // 2) + num_val // 2]) , 
                                                          np.array(slides_C [fold * (num_val // 2): fold * (num_val // 2) + num_val // 2])))
    return validation_slide_split.astype('int')
    
    
# In[2] Customized Dataset Class
# Returns a (transformed image, binary label, slide number) tuple for every item
# BM(Brain Metastatic) and C(Control) corresponds to Met+ and Met-, respectively
class NSCLC_Dataset(Dataset):
    def __init__(self, datafolder, datatype, transform, validation_slides):
        self.datafolder = datafolder
        self.image_files_list = {}
        self.categories = ['BM', 'C']
        self.dataset_image = []
        self.count = {}

        for category in self.categories:
            label = 0 if category == 'BM' else 1
            self.image_files_list[category] = [os.listdir(datafolder + s + '/') for s in self.categories]

            for img in self.image_files_list[category][label]:
                row_idx = int(np.floor( float(int(img[:-4])-1) / tile_per_slide))
                slide = row_idx + 1
                if datatype == 'train' and slide not in validation_slides:
                    if not slide in self.count:
                        self.count[slide] = 1
                    else:
                        self.count[slide] += 1
                    self.dataset_image.append((img, label, slide))
                elif datatype == 'val' and slide in validation_slides:
                    if not slide in self.count:
                        self.count[slide] = 1
                    else:
                        self.count[slide] += 1
                    self.dataset_image.append((img, label, slide))
        self.transform = transform
        

    def __len__(self):
        return len(self.dataset_image)

    def __getitem__(self, idx):
        if self.dataset_image[idx][1] == 0:
            img_name = os.path.join(self.datafolder + 'BM/',
                                    self.dataset_image[idx][0])
        elif self.dataset_image[idx][1] == 1:
            img_name = os.path.join(self.datafolder + 'C/',
                                    self.dataset_image[idx][0])
                
        image = Image.open(img_name)
        image = self.transform(image)
        return image, self.dataset_image[idx][1], self.dataset_image[idx][2]
    
    
# In[3] Main
if __name__ == '__main__':
    # Paths to where the dataset folder is stored 
    ''' CHANGE THIS '''
    Cpath = ''
    
    # Hyper Parameters and Paths
    root_name = 'NSCLC_Dataset' 
    model_abbr = 'Resenet18_' # Model Identifier
    magnif = '20'             # 20x Magnification Images
    num_epochs_list = [16,8,23] # Number of epochs to train for each train-test experiment, determined from cross-validation


    tile_per_slide = 1000
    
    # Train-test experiments
    nfold = 3    # Number of train-test experiments
    num_val = 40 # Number of slides reserved for testing in each train-test experiment
    
    # Model Training Parameters
    batch_size = 200    # Batch size
    num_workers = 2     # Number of workers for data loading
    lr = 1e-3           # Learning rate
    momentum = 0.9      # Momentum
    weight_decay = 0.1  # Weight decay
    
    data_dir = Cpath + root_name 
    indexpath = os.path.join(data_dir, 'Index')
    save_dir = os.path.join(data_dir, 'test_results')
    
    # Storing the final tile and slide level accuracies on each fold of cross validation
    nfold_val_tile_acc =  []
    nfold_val_slide_acc = []
    nfold_val_tile_auc =  []
    nfold_val_slide_auc = []
    
    
    # Directory to store testing results
    os.makedirs(save_dir, exist_ok = True)
    
    # Combine the train and testing index files
    iminfo_train = pd.read_csv(os.path.join(indexpath, 'iminfo_train.csv'))
    iminfo_test = pd.read_csv(os.path.join(indexpath, 'iminfo_test.csv'))
    iminfo_list = pd.concat([iminfo_train, iminfo_test], ignore_index=True)
    iminfo_list = iminfo_list.sort_values(by = ['Slide', 'Index'])
    del iminfo_train, iminfo_test
    
    
    # Get the slide numbers for BM and C
    # In the code, BM (Brain Metastatic) means Met+ patients
    # C (Control) means Met- patients
    slides_BM = []
    slides_C  = []
    for idx in range(len(iminfo_list) // tile_per_slide):
        if iminfo_list['Class'][idx * tile_per_slide] == 0: 
            slides_BM.append(iminfo_list['Slide'][idx * tile_per_slide])
        elif iminfo_list['Class'][idx * tile_per_slide] == 1:
            slides_C.append(iminfo_list['Slide'][idx * tile_per_slide])
        else:
            print('Error')
    slides_BM = np.array(slides_BM)
    slides_C  = np.array(slides_C)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(device)
    
    # Train Test Splits
    validation_slide_splits = shuffle(slides_BM, slides_C, num_val = num_val, nfold = nfold, overlap = 0)
    # Store which slides are used for testing in each train-test experiment
    np.save(os.path.join(indexpath,'nfold_splits_nfold_'+ str(nfold) + '.npy'), validation_slide_splits)
    
    # In[]
    for fold in range(nfold): # For each train-test experiment
        num_epochs = num_epochs_list[fold]
        
        # From the index file, separate the tiles used for training and the tiles used for validation
        iminfo_train_cv = iminfo_list.drop(np.arange(0, len(iminfo_list), 1))
        iminfo_val_cv = iminfo_list.drop(np.arange(0, len(iminfo_list), 1))
        
        slide_num = np.zeros(len(iminfo_list)//tile_per_slide)
        for row_idx in range(len(slide_num)):
            slide_n = int(iminfo_list['Index'][row_idx*tile_per_slide][7:])
            slide_num[row_idx] = slide_n
            if slide_n in validation_slide_splits[fold]: 
                iminfo_val_cv = pd.concat([ iminfo_val_cv,iminfo_list.loc[row_idx*tile_per_slide : (row_idx+1)*tile_per_slide - 1] ])
            elif slide_n not in validation_slide_splits[fold]: 
                iminfo_train_cv = pd.concat([ iminfo_train_cv,iminfo_list.loc[row_idx*tile_per_slide : (row_idx+1)*tile_per_slide - 1] ])
            else:
                print('Error: Slide split issue!')
                
        slide_num = slide_num.astype('int')
                
        # Load the precalculated mean and variance of each RGB channel
        # These valus will be used for standardization of model input
        mean_r = np.mean(iminfo_train_cv['mean_r'])
        mean_g = np.mean(iminfo_train_cv['mean_g'])
        mean_b = np.mean(iminfo_train_cv['mean_b'])
        std_r = np.sqrt(np.mean(iminfo_train_cv['var_r']))
        std_g = np.sqrt(np.mean(iminfo_train_cv['var_g']))
        std_b = np.sqrt(np.mean(iminfo_train_cv['var_b']))
        
        print(mean_r,mean_g,mean_b,std_r,std_g,std_b)
        
        # Data transform
        # For training, random cropping, rotations and flippings are added as data augmentation
        # For testing, only center cropping will be used
        data_transforms = {
            'train': transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(90),
                                    transforms.ToTensor(),
                                    transforms.Normalize([mean_r,mean_g,mean_b], [std_r,std_g,std_b])
                                    ]),
            'val': transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([mean_r,mean_g,mean_b], [std_r,std_g,std_b])
                                    ]),
            }
        
        # Customized Dataset and corresponding Dataloader for current train-test experiment
        image_datasets = {x: NSCLC_Dataset(data_dir + '/train/', x, data_transforms[x], validation_slide_splits[fold]) for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val'] }
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        print(dataset_sizes)
           
        
        save_name = root_name + '_lr_' + str(lr) + '_model_' + model_abbr + 'fold_' + str(int(fold))
        
        # Initialize pretrained model
        if model_abbr == 'Resenet18_': 
           model_ft = models.resnet18(pretrained=True)
           num_ftrs = model_ft.fc.in_features
           model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2))  

        
        model_ft = model_ft.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Decay LR by a factor of 0.1 every xx epochs 
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=200, gamma=0.1)
        
        # Number of parameters
        num_params = sum(param.numel() for param in model_ft.parameters())
        print('  ')
        print('Magnification: ' + magnif + ' | Model: ' + model_abbr + ' | Learning Rate: ' + str(lr))
        print('Number of parameters: ',num_params)
        print('Total Number of Epochs : ', num_epochs)
           

        # Begin Training 
           
        since = time.time()
        
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        train_loss =  []
        val_loss = []
        train_acc = []
        val_acc = []
        for epoch in range(num_epochs):
            t = time.time()
            print('-' * 40)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            
            # Training Phase
            model_ft.train()
                    
            running_loss = 0.0
            running_corrects = 0
         
            # Iterate over data.
            for inputs, labels, _ in dataloaders['train']:
                 inputs = inputs.to(device)
                 labels = labels.to(device)
                 
                 optimizer_ft.zero_grad()

                 with torch.set_grad_enabled(True):
                     outputs = model_ft(inputs)
                     preds_score, preds_class = torch.max(outputs,1)
                     loss = criterion(outputs, labels)
                  
                     loss.backward()
                     optimizer_ft.step()
            
                 # Calculate Loss and Accuracy
                 running_loss += loss.item() * inputs.size(0)
                 running_corrects += torch.sum(preds_class == labels.data)
             
                 exp_lr_scheduler.step()
            
            epoch_loss = (running_loss / dataset_sizes['train'])
            epoch_acc = (running_corrects / dataset_sizes['train']).cpu()
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
         
         
            
            elapsed = time.time() - t
            print('Training Time per epoch: ',elapsed)
            print('{} Loss: {:.4f} Acc: {:.4f} '.format('Train', epoch_loss, epoch_acc))
            
            
        time_elapsed = time.time() - since
        model_wts = copy.deepcopy(model_ft.state_dict())
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


        # Save trained model
        torch.save(model_ft, os.path.join(save_dir, save_name + '_whole_model.pt'))

        
