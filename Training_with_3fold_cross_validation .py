"""
For each train-test experiment, perform cross-validation to determine the best set of training parameters

@author: Haowen Zhou and Siyu (Steven) Lin, Oct 17, 2023
"""

# In[0] Dependencies and Library Configurations
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import models, transforms

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import copy

from sklearn.metrics import roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')

cudnn.benchmark = True
plt.ion()   # interactive mode


# In[1] Utility Functions
def soft(x,y):
    ''' Softmax of x to [x,y]'''
    return 1/(1 + np.exp(y-x))


def sig(x):
    ''' Sigmoid Activation'''
    return 1/(1 + np.exp(-x))


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
# BM(Brain Metastatic, Met+, 0) and C(Control, Met-, 1)
class NSCLC_Dataset(Dataset):
    def __init__(self, datafolder, datatype, transform, validation_slides, testing_slides):
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
                if datatype == 'train' and slide not in validation_slides and slide not in testing_slides:
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
    Cpath = ' '
    
    # Hyper Parameters and Paths
    root_name = 'NSCLC_Dataset'   # Dataset directory name
    model_abbr = 'Resenet18_'     # Model Identifier
    magnif = '20'                 # 20x Magnification Images
    tile_per_slide = 1000

    # Cross-validation and Train-test experiments
    num_train_test_splits = 3  # Number of train-test experiment
    nfold = 3                  # Number of folds in cross-validation
    num_test = 40 # Number of slides reserved for testing in each train-test experiment
    num_val = 30  # Number of slides reserved for validation in each cross-validation
    
    
    # Model Training Parameters
    batch_size = 200            # Batch size
    num_workers = 2             # Number of CPU workers for data loading
    lr = 1e-3                   # Learning rate  
    momentum = 0.9              # Momentum
    num_epochs = 30             # Number of epochs
    weight_decay = 0.1          # Weight decay
    
    # Paths Setup
    data_dir = os.path.join(Cpath, root_name)   # Path to the dataset folder
    indexpath = os.path.join(data_dir, 'Index') # Path to the index folder
    
    # Storing the final tile and slide level accuracies on each fold of cross validation
    nfold_val_tile_acc =  []    # Tile-level accuracy
    nfold_val_slide_acc = []    # Slide-level accuracy
    nfold_val_tile_auc =  []    # Tile-level AUC
    nfold_val_slide_auc = []    # Slide-level AUC
    
    
    # Combine the train and testing index files
    iminfo_train = pd.read_csv(os.path.join(indexpath, 'iminfo_train.csv'))    # Index file for training
    iminfo_test = pd.read_csv(os.path.join(indexpath, 'iminfo_test.csv'))      # Index file for testing
    iminfo_list = pd.concat([iminfo_train, iminfo_test], ignore_index=True)    # Combine the two index files
    iminfo_list = iminfo_list.sort_values(by = ['Slide', 'Index'])             # Sort the index file by slide number and tile number
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
    
    # Train Test Splits
    validation_slide_splits = shuffle(slides_BM, slides_C, num_val = num_test, nfold = num_train_test_splits)   # Train-test splits
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Use GPU if available
    print(device)
    
    # In[] 
    for train_test_fold in range(num_train_test_splits): # For each train-test experiment
    
        # Directory to store results and models
        save_dir = os.path.join(data_dir, 'test_results_train_test_splits_' + str(train_test_fold))
        os.makedirs(save_dir, exist_ok = True)
        
        # Keep the testing slides aside, and get the remaining slides
        cv_slides_BM = []
        cv_slides_C  = []
        for slide in slides_BM:
            if slide not in validation_slide_splits[train_test_fold]:
                cv_slides_BM.append(slide)
                
        for slide in slides_C:
            if slide not in validation_slide_splits[train_test_fold]:
                cv_slides_C.append(slide) 
                
        cv_slides_BM = np.array(cv_slides_BM)
        cv_slides_C  = np.array(cv_slides_C)
        
        # Use the non-testing slides and generate cross-validation splits
        cv_slide_splits = shuffle(cv_slides_BM, cv_slides_C, num_val = num_val, nfold = nfold)
        
        for fold in range(nfold): # For each cross-validation fold
            # From the index file, separate the tiles used for training and the tiles used for validation
            iminfo_train_cv = iminfo_list.drop(np.arange(0, len(iminfo_list), 1))
            iminfo_val_cv = iminfo_list.drop(np.arange(0, len(iminfo_list), 1))
            slide_num = np.zeros(len(iminfo_list)//tile_per_slide)
            for row_idx in range(len(slide_num)):
                slide_n = int(iminfo_list['Index'][row_idx*tile_per_slide][7:])
                slide_num[row_idx] = slide_n
                if slide_n in cv_slide_splits[fold]: 
                    iminfo_val_cv = pd.concat([ iminfo_val_cv,iminfo_list.loc[row_idx*tile_per_slide : (row_idx+1)*tile_per_slide - 1] ])
                elif slide_n not in cv_slide_splits[fold] and slide_n not in validation_slide_splits[train_test_fold]: 
                    iminfo_train_cv = pd.concat([ iminfo_train_cv,iminfo_list.loc[row_idx*tile_per_slide : (row_idx+1)*tile_per_slide - 1] ])
    
            
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
            # For (cross-)validation, only center cropping will be used
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
            
            # Customized Dataset and corresponding Dataloader for current cross-validation fold
            image_datasets = {x: NSCLC_Dataset(data_dir + '/train/', x, data_transforms[x], cv_slide_splits[fold], validation_slide_splits[train_test_fold]) for x in ['train', 'val']}
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val'] }
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
            print(dataset_sizes)
               
            save_name = root_name + '_lr_' + str(lr) + '_model_' + model_abbr + 'fold_' + str(train_test_fold) +'_' + str(int(fold))
            
            # Initialize pretrained model
            if model_abbr == 'Resenet18_': 
               model_ft = models.resnet18(pretrained=True)
               num_ftrs = model_ft.fc.in_features
               model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2))   
    
    
            model_ft = model_ft.to(device)
            
            criterion = nn.CrossEntropyLoss()
            
            # Observe that all parameters are being optimized
            optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Decay LR by a factor of 0.1 every xx epochs 
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=200, gamma=0.1)
            
            # Number of parameters
            num_params = sum(param.numel() for param in model_ft.parameters())
            print('  ')
            print('Magnification: ' + magnif + ' | Model: ' + model_abbr + ' | Learning Rate: ' + str(lr))
            print('Number of parameters: ',num_params)
            print('Total Number of Epochs : ', num_epochs)
               
            # Begin Training and Cross-Validation
            since = time.time()
            
            best_model_wts = copy.deepcopy(model_ft.state_dict())
            best_acc = 0.0
            best_epoch = 0
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
             
                # Training phase in Every Epoch
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
                
                # Validation phase in every epoch
                t = time.time()
                model_ft.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels, _ in dataloaders['val']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    outputs = model_ft(inputs)
                    
                    preds_score, preds_class = torch.max(outputs,1)
                    loss = criterion(outputs, labels)
    
                    # Keep track of Loss and Accuracy
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds_class == labels.data)
                             
                epoch_loss = (running_loss / dataset_sizes['val'])
                epoch_acc = (running_corrects / dataset_sizes['val']).cpu()
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
    
                elapsed = time.time() - t
                print('Validation Time per epoch: ',elapsed)
                print('{} Loss: {:.4f} Acc: {:.4f} '.format(
                    'Validation', epoch_loss, epoch_acc))
                
                # Save the model with the best validation performance
                if epoch_acc > best_acc:
                    best_epoch = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model_ft.state_dict())
                    
      
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
           
            # load the model weights with the best performance on validation
            model_ft.load_state_dict(best_model_wts)
            
            # Save model
            torch.save(model_ft, os.path.join(save_dir, save_name + '_whole_model.pt'))
             
            # Save the model performance statistics throughout training
            np.save(os.path.join(save_dir,'Train_loss_' + save_name + '.npy'), np.array(train_loss))
            np.save(os.path.join(save_dir,'Train_acc_' + save_name + '.npy'), np.array(train_acc))
            np.save(os.path.join(save_dir,'Val_loss_' + save_name + '.npy'), np.array(val_loss))
            np.save(os.path.join(save_dir,'Val_acc_' + save_name + '.npy'), np.array(val_acc))
            np.save(os.path.join(save_dir,'Best_epoch_' + save_name + '.npy'), np.array(best_epoch))
            
            
            # Get the tile-level accuracy based on the best model
            # by feeding all the validation tiles to the model
            slide_accuracies = {}
            model_ft.eval()
            running_corrects = 0
            
            tile_y_true = np.array([])
            tile_y_pred = np.array([])
            
            for inputs, labels, slides in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                outputs = model_ft(inputs)
                preds_score, preds_class = torch.max(outputs,1)
                loss = criterion(outputs, labels)
                
                tile_scores = soft(outputs[:,0].detach().cpu().numpy(),outputs[:,1].detach().cpu().numpy())
                slide_nums = slides.detach().cpu().numpy()
                
                tile_y_pred = np.append(tile_y_pred, preds_class.detach().cpu().numpy())
                tile_y_true = np.append(tile_y_true, labels.detach().cpu().numpy())
                
                
                for i in range(np.size(tile_scores)):
                    slide = slide_nums[i]
                    if slide not in slide_accuracies:
                        slide_accuracies[slide] = []
                    slide_accuracies[slide].append(tile_scores[i])
                # statistics
                running_corrects += torch.sum(preds_class == labels.data)
             
            overall_val_accuracy = (running_corrects / dataset_sizes['val']).cpu().numpy()
            tile_auc_score = roc_auc_score(tile_y_true, tile_y_pred)
            nfold_val_tile_auc.append(tile_auc_score)
            
            # Show results
            fpr, tpr, thres = roc_curve(tile_y_true, tile_y_pred)
            plt.plot(fpr,tpr)
            plt.title('Fold ' + str(fold) + ' Tile-Level ROC Curve')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig(os.path.join(save_dir, 'Fold ' + str(fold) + ' Tile-Level ROC Curve.png'))
            plt.show()
            print('Fold ' + str(fold) + ' Tile-Level AUC Score ' + str(tile_auc_score))
            
            # Get the slide-level accuracy based on the best model
            # by pooling the tile-level scores for each slide
            count_slide_pred_correct = 0
            slide_y_true = []
            slide_y_pred = []
            
            df = pd.DataFrame.from_dict(slide_accuracies)
            df.to_csv(os.path.join(save_dir, save_name + '_tile_accuracies.csv'), index=True)
            
            
            for slide, tile_preds in slide_accuracies.items():
                slide_pred = np.median(np.array(tile_preds))
                slide_pred = np.sign(slide_pred - 0.5)
                gt = 0
                if slide in slides_BM:
                    gt = 1
                elif slide in slides_C:
                    gt = -1
                if slide_pred == gt:
                    count_slide_pred_correct += 1
                slide_y_true.append(gt)
                slide_y_pred.append(slide_pred)
                    
            slide_y_true = np.array(slide_y_true)
            slide_y_pred = np.array(slide_y_pred)
            slide_auc_score = roc_auc_score(slide_y_true, slide_y_pred)
            
            # Show results
            fpr, tpr, thres = roc_curve(slide_y_true, slide_y_pred)
            plt.plot(fpr,tpr)
            plt.title('Fold ' + str(fold) + ' Slide-Level ROC Curve')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig(os.path.join(save_dir, 'Fold ' + str(fold) + ' Slide-Level ROC Curve.png'))
            plt.show()
            print('Fold ' + str(fold) + ' Slide-Level AUC Score ' + str(slide_auc_score))
            
            # Save results of the model for this cross validation fold
                    
            nfold_val_slide_acc.append(count_slide_pred_correct / len(slide_accuracies))
            nfold_val_tile_acc.append(overall_val_accuracy)
            nfold_val_slide_auc.append(slide_auc_score)
            print('Overall slide level accuracy: ', count_slide_pred_correct / len(slide_accuracies))
            print('Overall tile level accuracy: ', overall_val_accuracy)
        
        # Summary of the full cross-validation outcome for this train-test split
        df = pd.DataFrame({'Tile Accuracy':  nfold_val_tile_acc,
                           'Slide Accuracy': nfold_val_slide_acc,
                           'Tile AUC':  nfold_val_tile_auc,
                           'Slide AUC': nfold_val_slide_auc
                           })
        df.to_csv(os.path.join(save_dir, 'nfold_validation_accuracy.csv'), index=True)
                        