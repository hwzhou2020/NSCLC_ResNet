"""
For each train-test experiment, train a model based on predetermined parameters

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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import copy
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

from PIL import Image

import warnings
warnings.filterwarnings('ignore')

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve, auc

cudnn.benchmark = True
plt.ion()   # interactive mode


# In[1]
# Utility Functions
def soft(x,y):
    ''' Softmax of x to [x,y]'''
    return 1/(1 + np.exp(y-x))


def sig(x):
    ''' Sigmoid Activation'''
    return 1/(1 + np.exp(-x))


def shuffle(slides_BM, slides_C, nfold = 1, num_val = 20, overlap = 0):
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
    np.random.seed(1)
    np.random.shuffle(slides_BM)
    np.random.shuffle(slides_C)
    for fold in range(nfold):
        validation_slide_split[fold][:] = np.concatenate((np.array(slides_BM[fold * (num_val // 2 - overlap): fold * (num_val // 2 - overlap) + num_val // 2]) , 
                                                          np.array(slides_C [fold * (num_val // 2 - overlap): fold * (num_val // 2 - overlap) + num_val // 2])))
    return validation_slide_split.astype('int')
        
    

# In[2] Customized Dataset Class
# Returns a (transformed image, binary label, slide number) tuple for every item
# BM(Brain Metastatic, Met+, 0) and C(Control, Met-, 1)
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
                elif datatype == 'test' and slide in validation_slides:
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
    root_name = 'NSCLC_Dataset'  # Dataset directory name
    model_abbr = 'Resenet18_'    # Model Identifier
    magnif = '20'                # 20x Magnification Images
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
    num_workers = 2     # Number of workers for data loading
    
    
    data_dir = Cpath + root_name 
    indexpath = data_dir  + '/' + 'Index/'
    save_dir = os.path.join(data_dir, 'test_results_Final')
    
    # Storing the final tile and slide level accuracies on each fold of cross validation
    nfold_val_tile_acc =  []
    nfold_val_slide_acc = []
    nfold_val_tile_auc =  []
    nfold_val_slide_auc = []
    
    
    # Directory to store testing results
    os.makedirs(save_dir, exist_ok = True)
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
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
    
    # Train Test Splits
    validation_slide_splits = shuffle(slides_BM, slides_C, num_val = num_val, nfold = nfold, overlap = 0)    
    
    for fold in range(nfold): # For each train-test experiment
        num_epochs = num_epochs_list[fold]
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
            'test': transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([mean_r,mean_g,mean_b], [std_r,std_g,std_b])
                                    ]),
            }
        
        # Customized Dataset and corresponding Dataloader for current train-test experiment
        image_datasets = {x: NSCLC_Dataset(data_dir + '/train/', x, data_transforms[x], validation_slide_splits[fold]) for x in ['test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['test'] }
        dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
        print(dataset_sizes)
           
        # Load trained models
        save_name = root_name + '_lr_' + str(lr) + '_model_' + model_abbr + 'fold_' + str(int(fold))
        model_name = 'NSCLC_3rd_TumorOnly_mag_20_color_Yes_for_shuffle_colornorm_combined_lr_0.001_model_Resenet18_fold_' + str(int(fold)) + '_whole_model.pt'
        model_ft = torch.load(os.path.join(save_dir,model_name))
      
        model_ft = model_ft.to(device)
        

        
        # Get the tile-level testing accuracy
        slide_accuracies_soft = {}
        slide_accuracies_sig = {}
        model_ft.eval()
        running_corrects = 0
        
        tile_y_true = np.array([])
        tile_y_pred = np.array([])
        
        for inputs, labels, slides in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            preds_score, preds_class = torch.max(outputs,1)
            
            tile_scores_soft = soft(outputs[:,0].detach().cpu().numpy(),outputs[:,1].detach().cpu().numpy())
            tile_scores_sig = sig(outputs[:,0].detach().cpu().numpy())
            slide_nums = slides.detach().cpu().numpy()
            
            tile_y_pred = np.append(tile_y_pred, tile_scores_sig) #preds_class.detach().cpu().numpy()
            tile_y_true = np.append(tile_y_true, labels.detach().cpu().numpy())
            
            
            for i in range(np.size(tile_scores_soft)):
                slide = slide_nums[i]
                if slide not in slide_accuracies_soft:
                    slide_accuracies_soft[slide] = []
                slide_accuracies_soft[slide].append(tile_scores_soft[i])
            
            for i in range(np.size(tile_scores_sig)):
                slide = slide_nums[i]
                if slide not in slide_accuracies_sig:
                    slide_accuracies_sig[slide] = []
                slide_accuracies_sig[slide].append(tile_scores_sig[i])      
        
            # statistics
            running_corrects += torch.sum(preds_class == labels.data)
         
        overall_val_accuracy = (running_corrects / dataset_sizes['test']).cpu().numpy()
        

        
        # Show results
        tile_auc_score = roc_auc_score(1-tile_y_true, tile_y_pred)
        fpr, tpr, thres = roc_curve(1-tile_y_true, tile_y_pred)
        plt.plot(fpr,tpr)
        plt.title('Fold ' + str(fold) + ' Tile-Level ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.savefig(os.path.join(save_dir, 'Fold ' + str(fold) + ' Tile-Level ROC Curve.png'))
        plt.show()
        print('Fold ' + str(fold) + ' Tile-Level AUC Score ' + str(tile_auc_score))
        
        np.save(os.path.join(save_dir, 'Tile_y_true_' + save_name + '.npy'),1 - tile_y_true)
        np.save(os.path.join(save_dir, 'Tile_y_pred_' + save_name + '.npy'),tile_y_pred)
        
        # Get the slide-level test accuracy
        count_slide_pred_correct = 0
        slide_y_true = []
        slide_y_pred = []
        
        
        df = pd.DataFrame.from_dict(slide_accuracies_soft)
        df.to_csv(os.path.join(save_dir, save_name + '_tile_accuracies_soft.csv'), index=True)
        
        df = pd.DataFrame.from_dict(slide_accuracies_sig)
        df.to_csv(os.path.join(save_dir, save_name + '_tile_accuracies_sig.csv'), index=True)
        
        for slide, tile_preds in slide_accuracies_sig.items():
            slide_pred_raw = np.median(np.array(tile_preds))
            # print(slide_pred_raw)
            slide_pred = np.sign(slide_pred_raw - 0.5)
            gt = 0
            if slide in slides_BM:
                gt = 1
            elif slide in slides_C:
                gt = -1
            # print('Prediction: ', slide_pred, 'GT: ', gt)
            if slide_pred == gt:
                count_slide_pred_correct += 1
            slide_y_true.append(gt)
            slide_y_pred.append(slide_pred_raw)
                
        slide_y_true = np.array(slide_y_true)
        slide_y_pred = np.array(slide_y_pred)
        slide_auc_score = roc_auc_score(slide_y_true, slide_y_pred)
        fpr, tpr, thres = roc_curve(slide_y_true, slide_y_pred)
        plt.plot(fpr,tpr)
        plt.title('Fold ' + str(fold) + ' Slide-Level ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        print('Fold ' + str(fold) + ' Slide-Level AUC Score ' + str(slide_auc_score))
        
                
        nfold_val_slide_acc.append(count_slide_pred_correct / len(slide_accuracies_soft))
        nfold_val_tile_acc.append(overall_val_accuracy)
        nfold_val_slide_auc.append(slide_auc_score)
        print('Overall slide level accuracy: ', count_slide_pred_correct / len(slide_accuracies_soft))
        print('Overall tile level accuracy: ', overall_val_accuracy)
    
       
    
