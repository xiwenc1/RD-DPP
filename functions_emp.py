# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:37:53 2022

@author: xiwenc
"""



import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from torch.nn import ModuleList
import torchsummary
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
import random
from itertools import combinations
import math,time
import sys
import gc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
from models import MobileNetV2
# items = list(range(6))
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
# for c in combinations(items, 2):
#     print(c)

import torchmetrics

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
print(f'use device: {DEVICE}')
def image_gau_noise(im,var): #0-16
    x = im.clone()
    x = x+torch.rand(x.shape)*var
   
    return x
    



class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(input_dim, output_dim)
       )
        # self.linear2 = torch.nn.Linear(10, output_dim)

    def forward(self, x):
        # feat = self.linear1(x)
        # print(feat.shape)
        outputs = self.linear1(x)
        return outputs,x







def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix)) #shape为(item_size,)
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items






# ======================================================================================================



def estimate_diversity_old(model,data_loader,device = DEVICE ):
    model.eval()
    diversity = np.zeros(10)
    
    for i in range(10):
        locals()['class_feats_'+str(i)] = torch.tensor([]).to(device)
    
    
    
    for index,(x,y) in enumerate(data_loader):
        # if index == 0:
        #   print(x[0][0][14][14])

        x = x.to(device)
        # output = self.model(x) if allowed_classes is None else self.model(x)[:, allowed_classes]
        # x = x.reshape(-1,784).to(DEVICE)
        output,feats = model(x)
        
        for j in range(len(y)):
            
            i = y[j].item()
            
            locals()['class_feats_'+str(i)] = torch.cat((locals()['class_feats_'+str(i)],feats[j].unsqueeze(0)),0)
        
    
    # calcu diversity
    for i in range(10):
        feat_space = locals()['class_feats_'+str(i)]
        
        # print(feat_space.shape)
        # print(torch.var(feat_space,axis = 0).shape)
        
        items = list(range(len(feat_space)))
        
        count = 0
        for c in combinations(items, 2):
            count =  count +1
            ix1,ix2 = c  # each combination 
            # diversity[i] = diversity[i]+ (torch.sum((feat_space[ix1]-feat_space[ix2])**2)).item()
            diversity[i] = diversity[i]+ (F.cosine_similarity(feat_space[ix1],feat_space[ix2],axis=-1)).item()
        
        
        
        
        
        
        diversity[i] = diversity[i]/count  # mean value of diversity
        # diversity[i] =torch.sum(torch.var(feat_space,axis = 0)).item()
        
    return diversity







def estimate_diversity_old2(model,data_loader,device = DEVICE ):
    model.eval()
    diversity = np.zeros(10)
    
    for i in range(10):
        locals()['class_feats_'+str(i)] = torch.tensor([]).to(device)
    
    
    
    for index,(x,y) in enumerate(data_loader):
        # if index == 0:
        #   print(x[0][0][14][14])

        x = x.to(device)
        # output = self.model(x) if allowed_classes is None else self.model(x)[:, allowed_classes]
        # x = x.reshape(-1,784).to(DEVICE)
        output,feats = model(x)
        
        for j in range(len(y)):
            
            i = y[j].item()
            
            locals()['class_feats_'+str(i)] = torch.cat((locals()['class_feats_'+str(i)],feats[j].unsqueeze(0)),0)
        
    
    # calcu diversity
    for i in range(10):
        feat_space = locals()['class_feats_'+str(i)]
        
        # print(feat_space.shape)
        # print(torch.var(feat_space,axis = 0).shape)
        
        items = list(range(len(feat_space)))
        similarity_m = torch.ones((len(feat_space),len(feat_space))).to(device)
        count = 0
        for c in combinations(items, 2):
            count =  count +1
            ix1,ix2 = c  # each combination 
            # diversity[i] = diversity[i]+ (torch.sum((feat_space[ix1]-feat_space[ix2])**2)).item()
            similarity_m[ix1,ix2] = F.cosine_similarity(feat_space[ix1],feat_space[ix2],axis=-1)
            similarity_m[ix2,ix1] = similarity_m[ix1,ix2]
            
            
            
            # diversity[i] = diversity[i]+ (F.cosine_similarity(feat_space[ix1],feat_space[ix2],axis=-1)).item()
        
        
        
        
        
        
        diversity[i] = torch.linalg.det(similarity_m)  # mean value of diversity
        # diversity[i] =torch.sum(torch.var(feat_space,axis = 0)).item()
    print(diversity)
    return diversity



def estimate_diversity_old2(model,data_loader,device = DEVICE ):
    model.eval()
    diversity = np.zeros(10)
    
    for i in range(10):
        locals()['class_feats_'+str(i)] = torch.tensor([]).to(device)
    
    
    
    for index,(x,y) in enumerate(data_loader):
        # if index == 0:
        #   print(x[0][0][14][14])

        x = x.to(device)
        # output = self.model(x) if allowed_classes is None else self.model(x)[:, allowed_classes]
        # x = x.reshape(-1,784).to(DEVICE)
        output,feats = model(x)
        
        for j in range(len(y)):
            
            i = 0
            
            locals()['class_feats_'+str(i)] = torch.cat((locals()['class_feats_'+str(i)],feats[j].unsqueeze(0)),0)
        
    
    # calcu diversity
    for i in range(1):
        feat_space = locals()['class_feats_'+str(i)]
        
        # print(feat_space.shape)
        # print(torch.var(feat_space,axis = 0).shape)
        
        items = list(range(len(feat_space)))
        
        # count = 0
        # for c in combinations(items, 2):
        #     count =  count +1
        #     ix1,ix2 = c  # each combination 
        #     # diversity[i] = diversity[i]+ (torch.sum((feat_space[ix1]-feat_space[ix2])**2)).item()
        #     diversity[i] = diversity[i]+ (F.cosine_similarity(feat_space[ix1],feat_space[ix2],axis=-1)).item()
        
        
        
        
        
        
        # diversity[i] = diversity[i]/count  # mean value of diversity
        diversity[i] =torch.sum(torch.var(feat_space,axis = 0)).item()
        
    return diversity



class CNN(nn.Module):
    def __init__(self,w,h,in_channels=3,num_hidden=2,num_class=10):
        super(CNN,self).__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               #(in,w,h) --> (8,w,h)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)      #(8,w,h) --> (16,w/2,h/2)
        )
        self.encoder_list_1 = ModuleList([nn.Sequential(
            nn.Conv2d(
                in_channels=8*2**i,
                out_channels=8*2**(i+1),
                kernel_size=3,
                stride=1,
                padding=1
            ),                              
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)      
        ) for i in range(num_hidden)])
        
        # self.BN = nn.BatchNorm1d(288)
        
        self.output = nn.Linear(8*2**num_hidden*(w//(2**(num_hidden+1)))*(h//(2**(num_hidden+1)))  ,num_class)
        
    def forward(self, x):
        out = self.conv1(x) 
#         print(out.shape)
        for encoder in self.encoder_list_1:
            out = encoder(out)
            
        
#         out = self.conv2(out)                #
        
        feat = out.view(out.size(0),-1)       #
        # feat = self.BN(feat)
        # print(out.shape)
        out = self.output(feat)
        return out,feat




@torch.no_grad()
def test(model,val_loader):
    model.eval()
    
    acc_test_loss_list = []
    for step,(x,y) in enumerate(val_loader):
        x = x.to(DEVICE)
        # x = x.reshape(-1,784).to(DEVICE)
        y = y.to(DEVICE)
        
#         loss = loss_func(output,y)
#         Epoch_loss_list.append(loss.item())

#         optimizer.zero_grad() # clear grad
#         loss.backward()
#         optimizer.step()
        # acc on train dataset
        test_output,_ = model(x)
        pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
        accuracy = float((pred_y == y.cpu().detach().numpy()).astype(int).sum()) / float(y.size(0))
        acc_test_loss_list.append(accuracy)
        
    return np.mean(acc_test_loss_list)    
    
# @torch.no_grad()
# def test(model,val_loader):  # return AUC
#     model.eval()
    
#     acc_test_loss_list = []
#     for step,(x,y) in enumerate(val_loader):
#         x = x.to(DEVICE)
#         # x = x.reshape(-1,784).to(DEVICE)
#         y = y.to(DEVICE)
        
# #         loss = loss_func(output,y)
# #         Epoch_loss_list.append(loss.item())

# #         optimizer.zero_grad() # clear grad
# #         loss.backward()
# #         optimizer.step()
#         # acc on train dataset
#         test_output,_ = model(x)
        
#         p= torch.nn.functional.softmax(test_output, dim=1)
        
        
        
#         # pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
#         # accuracy = float((pred_y == y.cpu().detach().numpy()).astype(int).sum()) / float(y.size(0))
#         # F1 = torchmetrics.classification.F1Score(task="multiclass",num_classes=int(torch.max(y).item())+1).to(DEVICE)
#         E = torchmetrics.classification.AUROC(task="multiclass",num_classes=int(torch.max(y).item())+1).to(DEVICE)
#         # auroc
#         score = E(p, y).item()
        
#         acc_test_loss_list.append(score)
        
#     return np.mean(acc_test_loss_list)    


def train_cifar(train_loader,val_loader,DEVICE,EPOCH,model,optimizer,loss_func,scheduler=None):
    
    Model_list = []
    Loss_train_1 = []
    acc_train_1  = []
    acc_test_1  = []
    for epoch in range(EPOCH):
        model.train()
        Epoch_loss_list = []
        acc_train_loss_list = []
        acc_test_loss_list = []
        for step,(x,y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            output,_ = model(x)
            loss = loss_func(output,y)
            Epoch_loss_list.append(loss.item())

            optimizer.zero_grad() # clear grad
            loss.backward()
            optimizer.step()
            # acc on train dataset
            test_output,_ = model(x)
            pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
            accuracy = float((pred_y == y.cpu().detach().numpy()).astype(int).sum()) / float(y.size(0))
            acc_train_loss_list.append(accuracy)


    #         # acc on test dataset
    #         test_output = cnn1(test_x.to(DEVICE))

    #         # test_output, last_layer = cnn(test_x)
    #         pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
    #         accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    #         acc_test_loss_list.append(accuracy)



        #colloect loss and accuracy
        Epoch_loss = np.mean(Epoch_loss_list)  
        Loss_train_1.append(Epoch_loss) 
        Epoch_train_acc = np.mean(acc_train_loss_list)
        acc_train_1.append(Epoch_train_acc)
    #     Epoch_test_acc = np.mean(acc_test_loss_list)
          
        Epoch_test_acc= test(model,val_loader)
        acc_test_1.append(Epoch_test_acc)  
        
        print(f'Epoch:{epoch + 1}/{EPOCH}   loss:{Epoch_loss} Train acc:{Epoch_train_acc} Test acc:{Epoch_test_acc} ')
        
        if scheduler is not None:
            scheduler.step()
        


    # torch.save(cnn,'cnn_minist.pkl')
    print('finish training')
    
    return acc_train_1,acc_test_1

def train(train_loader,val_loader,DEVICE,EPOCH,model,optimizer,loss_func):
    model.train()
    Model_list = []
    Loss_train_1 = []
    acc_train_1  = []
    acc_test_1  = []
    for epoch in range(EPOCH):
        Epoch_loss_list = []
        acc_train_loss_list = []
        acc_test_loss_list = []
        for step,(x,y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            output,_ = model(x)
            loss = loss_func(output,y)
            Epoch_loss_list.append(loss.item())

            optimizer.zero_grad() # clear grad
            loss.backward()
            optimizer.step()
            # acc on train dataset
            test_output,_ = model(x)
            pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
            accuracy = float((pred_y == y.cpu().detach().numpy()).astype(int).sum()) / float(y.size(0))
            acc_train_loss_list.append(accuracy)


    #         # acc on test dataset
    #         test_output = cnn1(test_x.to(DEVICE))

    #         # test_output, last_layer = cnn(test_x)
    #         pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
    #         accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    #         acc_test_loss_list.append(accuracy)



        #colloect loss and accuracy
        Epoch_loss = np.mean(Epoch_loss_list)  
        Loss_train_1.append(Epoch_loss) 
        Epoch_train_acc = np.mean(acc_train_loss_list)
        acc_train_1.append(Epoch_train_acc)
    #     Epoch_test_acc = np.mean(acc_test_loss_list)
          
        Epoch_test_acc= test(model,val_loader)
        acc_test_1.append(Epoch_test_acc)  
        
        # print(f'Epoch:{epoch + 1}/{EPOCH}   loss:{Epoch_loss} Train acc:{Epoch_train_acc} Test acc:{Epoch_test_acc} ')
        print(f'Epoch:{epoch + 1}/{EPOCH}   loss:{Epoch_loss}  Test auc:{Epoch_test_acc} ')


    # torch.save(cnn,'cnn_minist.pkl')
    print('finish training')
    
    return acc_train_1,acc_test_1

def only_train(train_loader,DEVICE,EPOCH,model,optimizer,loss_func):
    model.train()
    Model_list = []
    Loss_train_1 = []
    acc_train_1  = []
    acc_test_1  = []
    for epoch in range(EPOCH):
        Epoch_loss_list = []
        acc_train_loss_list = []
        acc_test_loss_list = []
        for step,(x,y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            output,_ = model(x)
            loss = loss_func(output,y)
            Epoch_loss_list.append(loss.item())

            optimizer.zero_grad() # clear grad
            loss.backward()
            optimizer.step()
            # acc on train dataset
            test_output,_ = model(x)
            pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
            accuracy = float((pred_y == y.cpu().detach().numpy()).astype(int).sum()) / float(y.size(0))
            acc_train_loss_list.append(accuracy)


    #         # acc on test dataset
    #         test_output = cnn1(test_x.to(DEVICE))

    #         # test_output, last_layer = cnn(test_x)
    #         pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
    #         accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    #         acc_test_loss_list.append(accuracy)



        #colloect loss and accuracy
        Epoch_loss = np.mean(Epoch_loss_list)  
        Loss_train_1.append(Epoch_loss) 
        Epoch_train_acc = np.mean(acc_train_loss_list)
        acc_train_1.append(Epoch_train_acc)
    #     Epoch_test_acc = np.mean(acc_test_loss_list)
          
        # Epoch_test_acc= test(model,val_loader)
        # acc_test_1.append(Epoch_test_acc)  
        
        # print(f'Epoch:{epoch + 1}/{EPOCH}   loss:{Epoch_loss} Train acc:{Epoch_train_acc} ')
        


    # torch.save(cnn,'cnn_minist.pkl')
    print('finish training')
    
    return acc_train_1,acc_test_1


def train_dis(train_loader,val_loader,DEVICE,EPOCH,model,optimizer,loss_func,val_X,val_Y,alpha = 0.1):
    model.train()
    Model_list = []
    Loss_train_1 = []
    acc_train_1  = []
    acc_test_1  = []
    for epoch in range(EPOCH):
        Epoch_loss_list = []
        acc_train_loss_list = []
        acc_test_loss_list = []
        for step,(x,y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            output,_ = model(x)
            test_out1,_ = model(val_X.to(DEVICE))
            loss = loss_func(output,y)+alpha*loss_func(test_out1,val_Y.to(DEVICE))
            Epoch_loss_list.append(loss.item())

            optimizer.zero_grad() # clear grad
            loss.backward()
            optimizer.step()
            # acc on train dataset
            test_output,_ = model(x)
            pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
            accuracy = float((pred_y == y.cpu().detach().numpy()).astype(int).sum()) / float(y.size(0))
            acc_train_loss_list.append(accuracy)


    #         # acc on test dataset
    #         test_output = cnn1(test_x.to(DEVICE))

    #         # test_output, last_layer = cnn(test_x)
    #         pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
    #         accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    #         acc_test_loss_list.append(accuracy)



        #colloect loss and accuracy
        Epoch_loss = np.mean(Epoch_loss_list)  
        Loss_train_1.append(Epoch_loss) 
        Epoch_train_acc = np.mean(acc_train_loss_list)
        acc_train_1.append(Epoch_train_acc)
    #     Epoch_test_acc = np.mean(acc_test_loss_list)
          
        Epoch_test_acc= test(model,val_loader)
        acc_test_1.append(Epoch_test_acc)  
        
        print(f'Epoch:{epoch + 1}/{EPOCH}   loss:{Epoch_loss} Train AUROC:{Epoch_train_acc} Test AUROC:{Epoch_test_acc} ')
        


    # torch.save(cnn,'cnn_minist.pkl')
    print('finish training')
    
    return acc_train_1,acc_test_1



def image_mask_noise(im,index_mask): #0-16
    x = im.clone()
    if len(x.shape) == 4:
        B,C,W,H = x.shape
    elif  len(x.shape) == 3:
        B,W,H = x.shape
    else:
        B=1
        W,H = x.shape
        
    index_list= list(np.arange(16))
    len_block = W//4
    # index_mask = random.sample(index_list,mask_patch)
    for i in range(B):
        
        
        # print(index_mask)
        for j in index_mask:
            col = j%4
            row = j//4
            if len(x.shape) == 4:
                x[i,:,len_block*col:len_block*(col+1),len_block*row:len_block*(row+1)]=1e-8
            elif  len(x.shape) == 3:
                x[i,len_block*col:len_block*(col+1),len_block*row:len_block*(row+1)]=1e-8
            else:
                x[len_block*col:len_block*(col+1),len_block*row:len_block*(row+1)]=1e-8
                
    return x
    
    
    

class Logistic(nn.Module):
    def __init__(self,classes_num=10):
        super(Logistic, self).__init__()
        
        self.logic = nn.Sequential( 
            nn.Linear(28*28*1, classes_num)
        )
        
        
    def forward(self,x):
        x = x.view(-1, 28*28*1)
        x = self.logic(x)
        feat = 0
        return x,feat  

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            # nn.Linear(512, 10)
        )
        
        self.out = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        feat = x
        x = self.out(x)
        # x = nn.functional.normalize(x)
        return x,feat



def code_diversity(feat,y,num_class=10,epsilon= 0.7):
        feat = feat+1e-8
        m=len(feat)
        feat = feat.reshape(-1,len(feat))  #dxN
        feat= feat/torch.linalg.norm(feat, axis=1, keepdims=True)
        
        d = len(feat)
        
        R = 1/2*torch.logdet(torch.eye(d).to(DEVICE)+d/(m*epsilon**2)*feat@feat.T)
        # torch.logdet(torch.eye(d)+d/(m*epsilon**2)*feat@feat.T)
        R_c = 0
        for i in range(num_class):
            feat_c = feat[:,y==i]
            n_c = sum(y==i)
            R_c = R_c+ n_c/m *1/2*torch.logdet(torch.eye(d).to(DEVICE)+d/(n_c*epsilon**2+1e-8)*feat_c@feat_c.T)
    
        return R,R_c
    



def code_diversity_empirical_old(dataloader,model,num_class=10,epsilon= 0.7):
        
    R_list = []
    R_C_list = []
    
    for step,(x,y) in enumerate(dataloader):
        # if step ==0:
        #     break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        _,feat = model(x)

        feat = feat+1e-8
        m=len(feat)
        
        feat = feat.reshape(-1,len(feat))  #dxN
        feat= feat/torch.linalg.norm(feat, axis=1, keepdims=True)
        d = len(feat)
        
        
        R = 1/2*torch.logdet(torch.eye(d).to(DEVICE)+d/(m*epsilon**2)*feat@feat.T)
        R_list.append(R.item())
        # torch.logdet(torch.eye(d)+d/(m*epsilon**2)*feat@feat.T)
        R_c = 0
        for i in range(num_class):
            feat_c = feat[:,y==i]
            n_c = sum(y==i)
            R_c = R_c+ n_c/m *1/2*torch.logdet(torch.eye(d).to(DEVICE)+d/(n_c*epsilon**2+1e-8)*feat_c@feat_c.T)
            # print(R_c)
        R_C_list.append(R_c.item())
    R= np.mean(R_list)
    R_c = np.mean(R_C_list)
    return R,R_c



def code_diversity_empirical(dataloader,model,num_class=10,epsilon= 0.7):
        
    R_list = []
    R_C_list = []
    
    for step,(x,y) in enumerate(dataloader):
        # if step ==0:
        #     break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        _,feat = model(x)

        feat = feat+1e-8
        m=len(feat)
        
        feat = feat.reshape(-1,len(feat))  #dxN
        feat= feat/torch.linalg.norm(feat, axis=1, keepdims=True)
        d = len(feat)
        
        
        R = 1/2*torch.logdet(torch.eye(d).to(DEVICE)+d/(m*epsilon**2)*feat@feat.T)
        R_list.append(R.item())
        # torch.logdet(torch.eye(d)+d/(m*epsilon**2)*feat@feat.T)
        R_c = 0
        for i in range(num_class):
            feat_c = feat[:,y==i]
            n_c = sum(y==i)
            if n_c!=0:
                R_c = R_c+ n_c/m* torch.exp(1/2*torch.logdet(torch.eye(d).to(DEVICE)+d/(n_c*epsilon**2+1e-8)*feat_c@feat_c.T))
            # print(R_c)
        R_C_list.append(R_c.item())
    R= np.mean(R_list)
    R = np.exp(R)
    R_c = np.mean(R_C_list)
    return R,R_c


@torch.no_grad()
def feat_empirical(dataloader,model,num_class=10,feat_len = 288):
    model.eval()
    R_list = []
    R_C_list = []
    
    feat_all = torch.zeros(num_class,feat_len).to(DEVICE)
    coumt_all = torch.zeros(num_class).to(DEVICE)
    for step,(x,y) in enumerate(dataloader):
        # if step ==0:
        #     break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        _,feat = model(x)

        feat = feat+1e-8
        m=len(feat)
        
        feat = feat.reshape(-1,len(feat))  #dxN

        for i in range(num_class):
            
            feat_c = feat[:,y==i]
            n_c = sum(y==i)
            if n_c>0:
                coumt_all[i] = coumt_all[i]+n_c
                feat_all[i,:] = feat_all[i]+ torch.sum(feat_c,axis = 1)

    for i in range(num_class):
        if coumt_all[i]!=0:
            feat_all[i,:] = feat_all[i,:]/coumt_all[i]
        
    return feat_all.detach().cpu().numpy().reshape(-1)



    
    
def code_diversity_gain(feat,y,dataset_gt_X_temp,dataset_gt_Y_temp,num_class=10,epsilon= 0.7):
    Epsilon = epsilon
    # R,R_c = code_diversity(dataset_gt_X_temp,dataset_gt_Y_temp,num_class=10,epsilon=Epsilon)
    # d1 = R-R_c
    
    incre_x =  torch.cat((dataset_gt_X_temp, feat), 0)
    incre_y = torch.cat((dataset_gt_Y_temp, y), 0)
    
    Ri,R_ci = code_diversity(incre_x,incre_y,num_class=10,epsilon= Epsilon)
    d2 = Ri-R_ci
    diff = d2#-d1
    return diff 





