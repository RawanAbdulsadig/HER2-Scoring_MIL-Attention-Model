# -*- coding: utf-8 -*-
"""
@author: Rawan Abdulsadig
"""

import pickle
import pandas as pd
import time
import h5py
import torch
from torchvision import  models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold


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
    

def Train_Model(model , From, criterion , optimizer , n_epochs , bag_size, batch_size, filename, x_train,y_train,s_train,x_valid,y_valid,s_valid):
    '''
    
    Parameters
    ----------
    model : MILNetwork object
        initialized MIL model object
    From : string
        a string specifiying what type of source domain images the embedding model is transfered from, either 'ImageNet', 'PCAM' , 'IHC' or 'Random'
    criterion : loss function object
    optimizer : optim optimizer object
    n_epochs : int
        number of epochs
    bag_size : int
        number of patch images in a bag
    batch_size : int
        number of bags in a batch
    filename : string
        a file name to be used when saving the model parameters
    x_train : np.array
        training patch images
    y_train : np.array
        training patch image labels <patches are annotated with the parent slide labels>
    s_train : np.array
        training slide number/id
    x_valid : np.array
        validation patch images
    y_valid : np.array
        validation patch image labels <patches are annotated with the parent slide labels>
    s_valid : np.array
        validation slide number/id

    Returns
    -------
    training_loss : np.array
        the recorded training losses in each epoch
    validation_loss : np.array
        the recorded validation losses in each epoch
    training_acc : np.array
        the recorded training accuracies in each epoch
    validation_acc : np.array
        the recorded validation accuracies in each epoch

    '''

    min_valid_loss , max_valid_acc = np.Inf ,  0.0
    training_iterations = 100
    validation_iterations = 25
    valid_loss_min = min_valid_loss
    valid_acc_max = max_valid_acc
    training_loss , validation_loss = [], []
    training_acc , validation_acc = [] , []
    
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        
        ############
        # Training #
        ############
        y_true , y_pred = [] , []
        model.train()
        # preparing the transform object for training
        if From == 'Random':
            transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                            transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.8,1.2), shear=10, fillcolor=(245, 245, 245)),
                                            transforms.ToTensor()
                                            ,transforms.RandomApply([AddGaussianNoise(0., 0.1)], p=0.5)
                                            ])
        elif From == 'ImageNet':
            transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                            transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.8,1.2), shear=10, fillcolor=(245, 245, 245)),
                                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ,transforms.RandomApply([AddGaussianNoise(0., 0.1)], p=0.5)
                                            ])
        elif From == 'PCAM':
            transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                            transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.8,1.2), shear=10, fillcolor=(245, 245, 245)),
                                            transforms.ToTensor(), transforms.Normalize(mean=[1.0294, 0.5995, 1.3757], std=[0.8379, 1.0470, 0.8024])
                                            ,transforms.RandomApply([AddGaussianNoise(0., 0.1)], p=0.5)
                                            ])
        elif From == 'IHC':
            transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                            transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.8,1.2), shear=10, fillcolor=(245, 245, 245)),
                                            transforms.ToTensor(), transforms.Normalize(mean=[1.4128, 1.4503, 1.5884], std=[0.5338, 0.6164, 0.6477])
                                            ,transforms.RandomApply([AddGaussianNoise(0., 0.1)], p=0.5)
                                            ])
        
        for data, target in get_batch_of_bags(x_train, y_train, s_train, transform, bag_size , batch_size , training_iterations):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(output, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            data, target = data.cpu(), target.cpu() # to free up cuda memory
            _ , pred = _.cpu(), pred.cpu() # to free up cuda memory
        training_acc.append(np.sum(np.array(y_true) == np.array(y_pred)) / (training_iterations*batch_size))
        training_loss.append(train_loss/training_iterations)
        
        ##############
        # Validation #
        ##############
        y_true , y_pred = [] , []
        model.eval()
        # preparing the transform object for validation
        if From == 'Random':
            transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ToTensor()])
        elif From == 'ImageNet':
            transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif From == 'PCAM':
            transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ToTensor(),transforms.Normalize(mean=[1.0294, 0.5995, 1.3757], std=[0.8379, 1.0470, 0.8024])])
        elif From == 'IHC':
            transform = transforms.Compose([transforms.ToPILImage(mode= 'RGB'), transforms.ToTensor(),transforms.Normalize(mean=[1.4128, 1.4503, 1.5884], std=[0.5338, 0.6164, 0.6477])])

        for data, target in get_batch_of_bags(x_valid, y_valid, s_valid, transform, bag_size , batch_size , validation_iterations, seed=123):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target.long())
            valid_loss += loss.item()
            _, pred = torch.max(output, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            data, target = data.cpu(), target.cpu() # to free up cuda memory
            _, pred = _.cpu(), pred.cpu()
        validation_acc.append(np.sum(np.array(y_true) == np.array(y_pred))  / (validation_iterations*batch_size))
        validation_loss.append(valid_loss/validation_iterations)
        torch.save(model.state_dict(), filename+'.pt')
        if validation_loss[-1] <= valid_loss_min:
                torch.save(model.state_dict(), filename+'.pt')
                valid_loss_min = validation_loss[-1]
        # if validation_acc[-1] >= valid_acc_max:
        #        torch.save(model.state_dict(), filename+'.pt')
        #        valid_acc_max = validation_acc[-1]
                
    return training_loss , validation_loss , training_acc , validation_acc


def Print(text , filename):
    with open(filename, "a") as file_object:
        file_object.write(text)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    Folder = '/MIL-CrossValidation/'
    batch_size = 64
    bag_size = 100
    n_epochs = 50
    lr = 1e-03
    wd = 1e-07
    Target = 'Multiclass'               #Binary/Multiclass
    From = 'PCAM'                 # Random/ImageNet/PCAM/IHC
    itr = 1

    trial_name = '5FoldCV - MIL-Att - H&E-HER2 '+Target+' - AlexNet-Embedder -'+From+' - lr = '+str(lr)+' wd = '+str(wd)+' bag_size = '+str(bag_size)+' batch_size = '+str(batch_size)+' -'
    output_file = Folder+trial_name+' outputs.txt'
    # print(trial_name)
    Print(trial_name+'\n' , output_file)

    # importing the data
    X = h5py.File('/HER2_HE_96x96_training.h5', 'r')['x']
    Y = h5py.File('/HER2_HE_96x96_training.h5', 'r')['y']
    S = h5py.File('/HER2_HE_96x96_training.h5', 'r')['s']
    
    X , Y , S = np.array(X) ,np.array(Y), np.array(S)

    if Target == 'Binary':
        # constructing a binary classification dataset from the multiclass dataset
        Y[Y==1] = 0
        Y[Y==3] = 1
        Y[Y==2] = 1
        classes = [0 , 1]
        
    elif Target == 'Multiclass':
        classes = [0 , 1 , 2 , 3]


    Training_Loss, Training_Acc, Validation_Loss, Validation_Acc = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

    itr = 0
    slides = np.unique(S)
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    for train_index, valid_index in kf.split(slides):
        itr += 1
        boolidx = False
        for s in slides[train_index]:
            boolidx= np.logical_or(boolidx , np.array(S)==s)
        x_train = X[boolidx]
        y_train = Y[boolidx]
        s_train = S[boolidx]
        boolidx = False
        for s in slides[valid_index]:
            boolidx= np.logical_or(boolidx , np.array(S)==s)
        x_valid = X[boolidx]
        y_valid = Y[boolidx]
        s_valid = S[boolidx]
        Print('Starting '+trial_name+str(itr)+'...'+'\n' , output_file)
        # print('Starting '+trial_name+str(itr)+'...')

        if From == 'Random':
            AlexNet = models.alexnet(pretrained = False).features
        elif From == 'ImageNet':
            AlexNet = models.alexnet(pretrained = True).features
        elif From == 'PCAM':
            AlexNet = models.alexnet()
            n_inputs = AlexNet.classifier[6].in_features
            AlexNet.classifier[6] = nn.Linear(n_inputs, 2)
            AlexNet = nn.DataParallel(AlexNet)
            AlexNet.load_state_dict(torch.load('PCAM-pretrained_AlexNet.pt'))
            AlexNet = AlexNet.module.features
        elif From == 'IHC':
            AlexNet = models.alexnet()
            n_inputs = AlexNet.classifier[6].in_features
            AlexNet.classifier[6] = nn.Linear(n_inputs, 4)
            AlexNet = nn.DataParallel(AlexNet)
            AlexNet.load_state_dict(torch.load('IHC-pretrained_AlexNet.pt'))
            AlexNet = AlexNet.module.features
            
        for param in AlexNet.parameters():
            param.requires_grad = False
        
        AlexNet = AlexNet.cuda()
        startt = time.time()
        MIL = MILNetwork(AlexNet , bag_size , batch_size , len(classes)).cuda()
        optimizer = optim.Adam(MIL.parameters(), lr=lr, weight_decay = wd)
        criterion = nn.CrossEntropyLoss()
        training_loss , validation_loss , training_acc , validation_acc = Train_Model(MIL , From, criterion , optimizer , n_epochs , bag_size, batch_size,
                                                                                    Folder+trial_name+str(itr), x_train.copy(),y_train.copy(),s_train.copy(),x_valid.copy(),y_valid.copy(),s_valid.copy())
        endt = time.time()
        p = '('+str(round((endt-startt)/60))+ ' minutes) \t'+'Maximum validation accuracy was: '+str(max(validation_acc))+', and the minimum validation loss was: '+str(min(validation_loss))
        # print(p)
        Print(p+'\n' , output_file)

        Training_Loss['CV'+str(itr)] = training_loss
        Training_Acc['CV'+str(itr)] = training_acc
        Validation_Loss['CV'+str(itr)] = validation_loss
        Validation_Acc['CV'+str(itr)] = validation_acc

        del MIL
        del x_train
        del y_train
        del s_train
        del x_valid
        del y_valid
        del s_valid

    with open(Folder+trial_name+'_TraningProgresses.pkl', 'wb') as f:
        pickle.dump((Training_Loss, Training_Acc, Validation_Loss, Validation_Acc) , f)

  