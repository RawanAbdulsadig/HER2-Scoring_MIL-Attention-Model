# -*- coding: utf-8 -*-
"""
@author: Rawan Abdulsadig (kwsp174)
"""

import os
import pickle
import pandas as pd
import time
import h5py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def get_bag_of_instances(X_ , Y_ , S_ ,transform, bag_size, iterations):
    '''
    Parameters
    ----------
    X_ : np.array
        an array of image patches 
    Y_ : np.array
        an array of patch labels (patches extracted from the same slide image should have the same label corresponding to the slide HER2 score)
    S_ : np.array
        an array of slide numbers/ids corresponding to the patches
    transform : torchvision.transforms object
        a transforms object to apply image prerpocessing and augmentation
    bag_size : int
        the number of patch instances in the bag
    iterations : int
        the number of bags to be obtained when iterating through the function

    Yields
    ------
    a tuple of 2 torch tensors
        first element: 4-dimentional tensor representing the bag of transformed patch images
        second element: a tensor containing the label of the bag
    '''
    for _ in range(iterations):
        #Choosing a random slide number/id
        slids = np.unique(S_)
        s = np.random.choice(slids)
        #Obtaining the HER2 score of that slide
        label = np.max(Y_[S_==s]) # no critical reason for using max, they all should be the same anyway
        x = X_[S_==s] #filtering the patches corresponding to the selected slide
        x_bag = x[np.random.choice(x.shape[0], size=bag_size, replace=False)] #Obtaining a random sample of patch images from x to represent the bag, sampling without replacement
        x_tesnor_bag = torch.empty(size=(bag_size, 3, 96, 96))
        for i in range(bag_size):
            x_tesnor_bag[i] = transform(x_bag[i]) #preprocessing and augmenting the patches then storing them in a torch.tensor (torchvision.transforms only work on one image at a time)
        yield x_tesnor_bag , torch.tensor(label)

        
def get_batch_of_bags(X_, Y_, S_, transform, bag_size , batch_size , iterations , seed = None):
    '''

    Parameters
    ----------
    X_ : np.array
        an array of image patches 
    Y_ : np.array
        an array of patch labels (patches extracted from the same slide image should have the same label corresponding to the slide HER2 score)
    S_ : np.array
        an array of slide numbers/ids corresponding to the patches
    transform : torchvision.transforms object
        a transforms object to apply image prerpocessing and augmentation
    bag_size : int
        the number of patch instances in the bag
    batch_size : int
        the number of bags of patches in a batch
    iterations : int
        the number of batches to be obtained when iterating through the function
    seed : int, optional
        a random seed that can be set to reproduce the exact same batches of bags everytime the function is used, or can be set to None to obtain different batches of bags each time. The default is None.

    Yields
    ------
    batch_of_bags : 5-dimensional torch.tensor
        a batch of bags of patch images
    batch_of_labels: 1-dimentional torch.tensor
        a batch of labels corresponding to the bags

    '''
    np.random.seed(seed)
    for _ in range(iterations):
        batch_of_bags = torch.empty(size=(batch_size, bag_size, 3, 96, 96))
        batch_of_labels = torch.empty(size=(batch_size))
        for i,(images, label) in enumerate(get_bag_of_instances(X_, Y_, S_ , transform=transform
                                                              ,bag_size = bag_size , iterations = batch_size)):
                batch_of_bags[i] = images
                batch_of_labels[i] = label
        yield batch_of_bags , batch_of_labels
        

class AddGaussianNoise(object):
    '''
    To be used as an augmentation method within "transforms"
    '''
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class MILNetwork(nn.Module):
    def __init__(self , embeddeing_model , bag_size , batch_size , classes):
        '''
        Parameters
        ----------
        embeddeing_model : model object
            the pre-trained patch embedding model <AlexNet feature extraction layers in this case>
        bag_size : int
            the number of patch images in a bag
        batch_size : int
            the number of bags in a batch
        classes : int
            the number of target classes, 4 for HER2 grades or 2 for HER2 positive/negative

        '''
        super(MILNetwork, self).__init__()
        self.bag_size = bag_size
        self.batch_size = batch_size
        self.embedder = embeddeing_model
        self.fc1 = nn.Linear(1024, classes)
        self.attention =    nn.Sequential(
                                nn.Conv1d(1024,256,1),
                                nn.Tanh(),
                                nn.Conv1d(256,1,1))

    def forward(self, x):
        out = torch.empty(size=(self.batch_size, 1024, self.bag_size)).cuda()
        for i in range(x.shape[0]):
            x_ = self.embedder.forward(x[i])
            x_ = x_.view(-1,1024)
            out[i] = x_.permute(1,0)
        A = self.attention(out)
        A = F.softmax(A , dim =2)
        M = torch.bmm(A, torch.transpose(out , 2, 1))
        M = M.view(-1, 1024)
        out = self.fc1(M)
        return out


def Test_Model(model ,From , x_test , y_test , s_test, criterion , bag_size , batch_size ):
    '''
    
    Parameters
    ----------
    model : MILNetwork object
        Trained MIL model object
    From : string
        a string specifiying what type of source domain images the embedding model was transfered from, either 'ImageNet', 'PCAM' , 'IHC' or 'Random'
    x_test : np.array
        testing patch-images
    y_test : np.array
        testing patch-image labels <patches are annotated with the parent slide labels>
    s_test : np.array
        testing slide number/id
    criterion : loss function object
    bag_size : int
        number of patch-images in a bag
    batch_size : int
        number of bags in a batch

    Returns
    -------
    Tuple
        int test loss , np.array of actual scores, np.array of predicted scores , np.array of predicted propabilities

    '''
    if From == 'Random':
        transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ToTensor()])
    elif From == 'ImageNet':
        transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif From == 'PCAM':
        transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ToTensor(),transforms.Normalize(mean=[1.0294, 0.5995, 1.3757], std=[0.8379, 1.0470, 0.8024])])
    elif From == 'IHC':
        transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ToTensor(),transforms.Normalize(mean=[1.4128, 1.4503, 1.5884], std=[0.5338, 0.6164, 0.6477])])

    testing_iterations = 25
    softmax = nn.Softmax(dim=1)
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []
    model.eval()
    for data, target in get_batch_of_bags(x_test, y_test, s_test, transform, bag_size , batch_size , testing_iterations , seed=123):
        data , target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target.long())
        test_loss += loss.item()
        _, pred = torch.max(output , 1)
        data , target , pred = data.cpu() , target.cpu() , pred.cpu()
        y_true.extend(target.numpy())
        y_pred.extend(pred.numpy())
        y_probs.extend(softmax(output).cpu().detach().numpy())
    test_loss = test_loss/testing_iterations
    return test_loss, np.array(y_true) , np.array(y_pred), np.array(y_probs)


def Print(text , filename):
    with open(filename, "a") as file_object:
        file_object.write(text)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    Folder = '/MIL-CrossValidation/'
    batch_size = 64
    bag_size = 100
    lr = 1e-03
    wd = 1e-07
    Target = 'Multiclass'               #Binary/Multiclass
    From = 'PCAM'                 # Random/ImageNet/PCAM/IHC


    trial_name = '5FoldCV - MIL-Att - H&E-HER2 '+Target+' - AlexNet-Embedder -'+From+' - lr = '+str(lr)+' wd = '+str(wd)+' bag_size = '+str(bag_size)+' batch_size = '+str(batch_size)+' -'
    output_file = Folder+trial_name+' - outputs.txt'
    # print(trial_name)
    Print(trial_name+'\n' , output_file)

    # importing the data
    X = h5py.File('/HER2_HE_96x96_testing.h5', 'r')['x']
    Y = h5py.File('/HER2_HE_96x96_testing.h5', 'r')['y']
    S = h5py.File('/HER2_HE_96x96_testing.h5', 'r')['s']
    
    X , Y , S = np.array(X) ,np.array(Y), np.array(S)

    if Target == 'Binary':
        # constructing a binary classification dataset from the multiclass dataset
        Y[Y==1] = 0
        Y[Y==3] = 1
        Y[Y==2] = 1
        classes = [0 , 1]
        
    elif Target == 'Multiclass':
        classes = [0 , 1 , 2 , 3]


    Test_loss = []
    precision_curves , recall_curves = [] , []
    if Target == 'Multiclass' :
        Avg_Accuracies, Precisions_0, Precisions_1, Precisions_2, Precisions_3, Recalls_0, Recalls_1, Recalls_2, Recalls_3, F1_scores_0, F1_scores_1, F1_scores_2, F1_scores_3, AUCs_0, AUCs_1, AUCs_2, AUCs_3 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    elif Target == 'Binary':
        Avg_Accuracies, Precisions_0, Precisions_1, Recalls_0, Recalls_1, F1_scores_0, F1_scores_1, AUCs_0 , AUCs_1 = [],[],[],[],[],[],[],[],[]


    for i in range(1,6):
        x_test , y_test , s_test =  np.array(X), np.array(Y), np.array(S)
        AlexNet = models.alexnet().features
        AlexNet = AlexNet.cuda()
        startt = time.time()
        MIL = MILNetwork(AlexNet , bag_size , batch_size , len(classes)).cuda()
        criterion = nn.CrossEntropyLoss()

        MIL.load_state_dict(torch.load(Folder+trial_name+str(i)+'.pt'))
        test_loss, y_true, y_pred, y_probs = Test_Model(MIL ,From , x_test.copy() , y_test.copy() , s_test.copy(), criterion , bag_size , batch_size )

        Test_loss.append(test_loss)
        if Target == 'Multiclass' :
            Avg_Accuracies.append(metrics.balanced_accuracy_score(y_true, y_pred))
            Precisions = metrics.precision_score(y_true, y_pred , average=None)
            Precisions_0.append(Precisions[0])
            Precisions_1.append(Precisions[1])
            Precisions_2.append(Precisions[2])
            Precisions_3.append(Precisions[3])
            Recalls = metrics.recall_score(y_true, y_pred , average=None)
            Recalls_0.append(Recalls[0])
            Recalls_1.append(Recalls[1])
            Recalls_2.append(Recalls[2])
            Recalls_3.append(Recalls[3])
            F1_scores = metrics.f1_score(y_true, y_pred , average=None)
            F1_scores_0.append(F1_scores[0])
            F1_scores_1.append(F1_scores[1])
            F1_scores_2.append(F1_scores[2])
            F1_scores_3.append(F1_scores[3])
            y_test = np.array(pd.get_dummies(y_true))
            y_score = y_probs
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for j in range(len(classes)):
                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
                roc_auc[j] = auc(fpr[j], tpr[j])
            AUCs_0.append(roc_auc[0])
            AUCs_1.append(roc_auc[1])
            AUCs_2.append(roc_auc[2])
            AUCs_3.append(roc_auc[3])
            precision = []
            recall = []
            for j in range(len(classes)):
                prec, rec, _ = precision_recall_curve(y_test[:, j],y_score[:, j])
                precision.append(prec)
                recall.append(rec)
            precision_curves.append(precision)
            recall_curves.append(recall)

        elif Target == 'Binary':
            Avg_Accuracies.append(metrics.balanced_accuracy_score(y_true, y_pred))
            Precisions_0.append(metrics.precision_score(y_true, y_pred , pos_label= 0))
            Precisions_1.append(metrics.precision_score(y_true, y_pred , pos_label= 1))
            Recalls_0.append(metrics.recall_score(y_true, y_pred , pos_label= 0))
            Recalls_1.append(metrics.recall_score(y_true, y_pred , pos_label= 1))
            F1_scores_0.append(metrics.f1_score(y_true, y_pred , pos_label= 0))
            F1_scores_1.append(metrics.f1_score(y_true, y_pred , pos_label= 1))
            y_test = np.array(pd.get_dummies(y_true))
            y_score = y_probs
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for j in range(len(classes)):
                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
                roc_auc[j] = auc(fpr[j], tpr[j])
            AUCs_0.append(roc_auc[0])
            AUCs_1.append(roc_auc[1])
            precision = []
            recall = []
            for j in range(len(classes)):
                prec, rec, _ = precision_recall_curve(y_test[:, j],y_score[:, j])
                precision.append(prec)
                recall.append(rec)
            precision_curves.append(precision)
            recall_curves.append(recall)
        del MIL

    if Target == 'Multiclass' :
        precision_ = (Precisions_0, Precisions_1, Precisions_2, Precisions_3 , precision_curves)
        recall_ = (Recalls_0, Recalls_1, Recalls_2, Recalls_3 , recall_curves)
        f1_score_ = (F1_scores_0, F1_scores_1, F1_scores_2, F1_scores_3)
        auc_ = (AUCs_0, AUCs_1, AUCs_2, AUCs_3)
    elif Target == 'Binary':
        precision_ = (Precisions_0, Precisions_1 , precision_curves)
        recall_ = (Recalls_0, Recalls_1 , recall_curves)
        f1_score_ = (F1_scores_0, F1_scores_1)
        auc_ = (AUCs_0 , AUCs_1)

    with open(Folder+trial_name+'_TMs.pkl', 'wb') as f:
        pickle.dump((Test_loss, Avg_Accuracies, precision_, recall_, f1_score_, auc_) , f)
