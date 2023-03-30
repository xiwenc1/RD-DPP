# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:10:09 2023

@author: xiwenc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:43:59 2023

@author: xiwenc
"""


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:32'


import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
# from torch.nn import ModuleList
import torchsummary
import copy
import random
# from options import args_parser
from functions_emp import train,test, \
code_diversity_empirical,code_diversity_empirical_old,estimate_diversity_empirical,feat_empirical,dpp,image_gau_noise,train_cifar

from torch.utils.data import TensorDataset

import sys
# from data_split import mnist_noniid,DatasetSplit
import json
# import pynvml
# import psutil
import argparse

import scipy.io as scio
from models.efficientnet import EfficientNetB0


plt.close('all')

# eval_period_list = [3]

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--EPOCH', type=int, default=100, help="epoches of training")
    parser.add_argument('--index', type=int, default=0, help="index of filename")
    # parser.add_argument('--var', type=float, default=0.5, help="variance of noise")
    parser.add_argument('--BATCH_SIZE', type=int, default=200, help="batch size")
    parser.add_argument('--LR', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--Total_budget', type=int, default=60, help="Total budget")
    parser.add_argument('--T_BS', type=int, default=128, help="training batch size")
    parser.add_argument('--clean_rate', type=float, default=3/30, help="clean data rate")
 
    # model arguments


    # other arguments
    parser.add_argument('--dataset_name', type=str, default='CIFAR10', help="name of dataset")
    parser.add_argument('--name_index', type=int, default=1, help="name index")
    parser.add_argument('--repeat', type=int, default=10, help="number of repeat")
    parser.add_argument('--T', type=int, default=10, help="number of packets in each round")

    
    args = parser.parse_args()
    return args

args = args_parser()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Total_budget = args.Total_budget




multip_GPU = (torch.cuda.device_count()>1)

BATCH_SIZE = args.BATCH_SIZE

C_ini =5
num_feat = 320 # change based on the network


train_batch = args.T_BS

# load data

eval_period_list = [args.T]




path = f'exp_{args.T}/'
os.makedirs(path,exist_ok=True)

if args.dataset_name == 'CIFAR10':
    
    
    Tcifar1 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
    
    
    train_data = torchvision.datasets.CIFAR10(
        root = 'data/CIFAR10',  #
        train = True,       #False: tEST
        # transform = torchvision.transforms.ToTensor(),
        transform=Tcifar1,#norm
        download=True
    )
    
    print(train_data.data.shape)
    print(len(train_data.targets))
    # train_data.data = torch.tensor(train_data.data)
    # train_data.targets = torch.tensor((train_data.targets))
    Tcifar2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    testset = torchvision.datasets.CIFAR10(
    root='data/CIFAR10', train=False, download=True, transform=Tcifar2)

    
    # testset.data = torch.tensor(testset.data)
    # testset.targets = torch.tensor((testset.targets))
    
    
    
    
    #
    num_class = 10
    




with open('data/'+args.dataset_name+'.txt') as f:  #load cluster result
        data_index = f.read()
        
dict_users = json.loads(data_index)


all_len =len(dict_users)
Noise_array = np.zeros((all_len,10),dtype=object)


for run_index in range(args.repeat):


    
    
    dataset_pre_X = torch.tensor([]) #candidate clusters data
    dataset_pre_Y = torch.tensor([])
    
       
    dataset_gt_X_temp = torch.tensor([])  #initial data
    dataset_gt_Y_temp = torch.tensor([])
    
    
    dataset_pre_X_all = torch.tensor([]) #all data
    dataset_pre_Y_all = torch.tensor([])
    
    
    all_index_cout = 0
    init_index = []
    
    loader_all_data = torch.utils.data.DataLoader(
        train_data, batch_size=len(train_data.targets), shuffle=False, num_workers=0)
    
    for (X_train,Y_train) in loader_all_data:
        pass
    
    
    
    
    
    for step in range(all_len):
        idxs = dict_users[str(step)]
        rest = len(idxs)%BATCH_SIZE
        idxs = idxs[:len(idxs)-rest]
        x= X_train[idxs]
        y =Y_train[idxs]
        # x = x.unsqueeze(1)
        data_loader_for_split = Data.DataLoader(TensorDataset(x,y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
    
        for index_pre, (x_,y_) in enumerate(data_loader_for_split):
            
            index_mask = Noise_array[step,0]
                #noisy_base = image_mask_noise(x.clone().detach(),index_mask)
            noisy_base = image_gau_noise(x_.clone().detach(),index_mask)
            noisy_base = torch.clip(noisy_base,0,1)
            if index_pre==0:
                if step <C_ini:
                    init_index.append(all_index_cout)
                    dataset_gt_X_temp = torch.cat((dataset_gt_X_temp, x_), 0)
                    dataset_gt_Y_temp = torch.cat((dataset_gt_Y_temp, y_), 0)
            # (x,y) = DatasetSplit(train_data, idxs)
                # index_mask = Noise_array[step,0]
                # #noisy_base = image_mask_noise(x.clone().detach(),index_mask)
                # noisy_base = image_gau_noise(x_.clone().detach(),index_mask)
                # noisy_base = torch.clip(noisy_base,0,1)
                dataset_pre_X = torch.cat((dataset_pre_X, noisy_base), 0)
                dataset_pre_Y = torch.cat((dataset_pre_Y, y_), 0)
            
            dataset_pre_X_all = torch.cat((dataset_pre_X_all, noisy_base), 0)
            dataset_pre_Y_all = torch.cat((dataset_pre_Y_all, y_), 0)
            all_index_cout = all_index_cout+1
            
        
    
    # train_loader_all= Data.DataLoader(TensorDataset(dataset_pre_X_all,dataset_pre_Y_all.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=False,pin_memory=True)
        
            
        
    train_loader= Data.DataLoader(TensorDataset(dataset_pre_X,dataset_pre_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=False,pin_memory=True)
        
    # train_loader_all= Data.DataLoader(TensorDataset(X_train,Y_train.type(torch.LongTensor)),batch_size=256,shuffle=True,pin_memory=True)
    
    val_loader = Data.DataLoader(dataset=testset,batch_size=BATCH_SIZE,shuffle=True)
    
    
        
    # model = EfficientNetB0()
    
    
    LR = args.LR
    LR0 =1e-4
    
    EPOCH =  args.EPOCH
    
    
    
    
    model_gt =EfficientNetB0().to(DEVICE)
    
    
        
    
    model_gt_temp =copy.deepcopy(model_gt).to(DEVICE)
    if multip_GPU:
        model_gt_temp = torch.nn.DataParallel(model_gt_temp)
    # model_gt_part= model(28,28,1,num_hidden=2).to(DEVICE)
    # model_noise =copy.deepcopy(model_gt).to(DEVICE)
    # model_base= copy.deepcopy(model_gt).to(DEVICE)
    # model_entropy=copy.deepcopy(model_gt).to(DEVICE)
    # model_rand = copy.deepcopy(model_gt).to(DEVICE)
    LR0 =1e-4
    optimizer_gt = torch.optim.SGD(model_gt.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
    # optimizer_gt_part = torch.optim.SGD(model_gt_part.parameters(),lr=LR)
    optimizer_gt_temp = torch.optim.SGD(model_gt_temp.parameters(),lr=LR0,
                      momentum=0.9, weight_decay=5e-4)
    # optimizer_noise = torch.optim.SGD(model_noise.parameters(),lr=LR)
    # optimizer_base = torch.optim.SGD(model_base.parameters(),lr=LR)
    # optimizer_entropy= torch.optim.SGD(model_entropy.parameters(),lr=LR)
    # optimizer_rand= torch.optim.SGD(model_rand.parameters(),lr=LR)
    # optimizer_base_2 = torch.optim.SGD(model_base_2.parameters(),lr=LR)
    
    loss_func = nn.CrossEntropyLoss()
    
    Dataloader_gt_temp= Data.DataLoader(TensorDataset(dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor)),batch_size=128,shuffle=True,pin_memory=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gt_temp, T_max=200)
    acc_train_gt_temp,acc_test_gt_temp = train_cifar(Dataloader_gt_temp,val_loader,DEVICE,EPOCH,model_gt_temp,optimizer_gt_temp,loss_func,scheduler)
    model_gt_temp.eval()
    
    model_eval_base = copy.deepcopy(model_gt_temp).to(DEVICE)
    model_eval_gt = copy.deepcopy(model_gt_temp).to(DEVICE)
    model_eval_entropy = copy.deepcopy(model_gt_temp).to(DEVICE)
    
    
    
    def pick_by_random(eval_period):
            acc_rand_list = np.zeros((Total_budget,2,EPOCH))
            dataset_rand_X = dataset_gt_X_temp
            dataset_rand_Y = dataset_gt_Y_temp
            # eval_period =10
            rand_index_list = random.sample(list(range(C_ini,len( train_loader))),Total_budget)
            for num_trans in range(0,Total_budget,eval_period):
                rand_index = rand_index_list[num_trans:num_trans+eval_period]
                print(rand_index)
                for step,(x,y) in enumerate(train_loader):
                    
                    if step in  rand_index:
                        # count = count+1
                        dataset_rand_X = torch.cat((dataset_rand_X, x), 0)
                        dataset_rand_Y = torch.cat((dataset_rand_Y, y), 0)
                
                if num_trans%eval_period ==0:
                    Dataloader_rand = Data.DataLoader(TensorDataset(dataset_rand_X,dataset_rand_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
                    model_rand = EfficientNetB0().to(DEVICE)
                    if multip_GPU:
                        model_rand = torch.nn.DataParallel(model_rand)
                    optimizer_rand= torch.optim.SGD(model_rand.parameters(),lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_rand, T_max=200)
                    acc_train_rand,acc_test_rand = train_cifar(Dataloader_rand,val_loader,DEVICE,EPOCH,model_rand,optimizer_rand,loss_func,scheduler)
                    
                    acc_rand_list[num_trans,0,:]=np.array(acc_train_rand)
                    acc_rand_list[num_trans,1,:]=np.array(acc_test_rand)
                    
            Dataloader_rand = Data.DataLoader(TensorDataset(dataset_rand_X,dataset_rand_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
            print(len(dataset_rand_X))
            model_rand = EfficientNetB0().to(DEVICE)
            if multip_GPU:
                        model_rand = torch.nn.DataParallel(model_rand)
            optimizer_rand= torch.optim.SGD(model_rand.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_rand, T_max=200)
            acc_train_rand,acc_test_rand = train_cifar(Dataloader_rand,val_loader,DEVICE,EPOCH,model_rand,optimizer_rand,loss_func,scheduler)
            return acc_rand_list,acc_test_rand,rand_index_list
    
    
    
    
    def pick_by_entropy(eval_period):  #uncertainty
            acc_entropy_list = np.zeros((Total_budget,2,EPOCH))
            dataset_model_entropy_X= dataset_gt_X_temp
            dataset_model_entropy_Y = dataset_gt_Y_temp
            model_eval_entropy = copy.deepcopy(model_gt_temp).to(DEVICE)
            pick_index1 = []
            for num_trans in range(0,Total_budget,eval_period):
                # print(num_trans)
                loss_gain_list = np.zeros(len(train_loader))
                
                for step,(x,y) in enumerate(train_loader):
                    if step>=C_ini and (step not in pick_index1):
                        # print(step)
                        # dataset_base_X_temp = torch.cat((dataset_base_X, x), 0)
                        # dataset_base_Y_temp = torch.cat((dataset_base_Y, y), 0)
                        
                        dataset_entropy_X_temp =x# torch.cat((dataset_gt_X_temp, x), 0)
                        dataset_entropy_Y_temp =y# torch.cat((dataset_gt_Y_temp, y), 0)
                        
                        # Dataloader_entropy_temp = Data.DataLoader(TensorDataset(dataset_entropy_X_temp,dataset_entropy_Y_temp.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                        
                        # loss =  1/loss_func(model_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                        loss =  loss_func(model_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                        loss_gain_list[step] = loss
                        
                        
                        # div,_ = estimate_diversity_empirical(model_eval_base,Dataloader_base_temp)
                        # marginal_gain_list[step] = div
                        
                index = np.argsort(loss_gain_list)[-eval_period:]
                # index = np.argsort(loss_gain_list)[0:eval_period]
                pick_index1.extend(index)
                
                for step,(x,y) in enumerate(train_loader):
                    if step in index:
                        dataset_model_entropy_X = torch.cat((dataset_model_entropy_X, x), 0)
                        dataset_model_entropy_Y = torch.cat((dataset_model_entropy_Y, y), 0)
                        # break
                
                if num_trans%eval_period ==0:
                    Dataloader_entropy = Data.DataLoader(TensorDataset(dataset_model_entropy_X,dataset_model_entropy_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
                    model_eval_entropy = EfficientNetB0().to(DEVICE)
                    if multip_GPU:
                        model_eval_entropy = torch.nn.DataParallel(model_eval_entropy)
                    optimizer_entropy = torch.optim.SGD(model_eval_entropy.parameters(),lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_entropy, T_max=200)
                    acc_train_e,acc_test_e = train_cifar(Dataloader_entropy,val_loader,DEVICE,EPOCH,model_eval_entropy,optimizer_entropy,loss_func,scheduler)
                    # acc_train_e,acc_test_e =train_dis(Dataloader_entropy,val_loader,DEVICE,EPOCH,model_eval_entropy,optimizer_entropy,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                    acc_entropy_list[num_trans,0,:]=np.array(acc_train_e)
                    acc_entropy_list[num_trans,1,:]=np.array(acc_test_e)
            Dataloader_entropy = Data.DataLoader(TensorDataset(dataset_model_entropy_X,dataset_model_entropy_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
            model_eval_entropy = EfficientNetB0().to(DEVICE)
            if multip_GPU:
                        model_eval_entropy = torch.nn.DataParallel(model_eval_entropy)
            optimizer_entropy = torch.optim.SGD(model_eval_entropy.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_entropy, T_max=200)
            acc_train_e,acc_test_e = train_cifar(Dataloader_entropy,val_loader,DEVICE,EPOCH,model_eval_entropy,optimizer_entropy,loss_func,scheduler)
            return acc_entropy_list,acc_test_e,pick_index1
    
    
    def pick_by_min_margin(eval_period):  #uncertainty
            acc_entropy_list = np.zeros((Total_budget,2,EPOCH))
            dataset_model_entropy_X= dataset_gt_X_temp
            dataset_model_entropy_Y = dataset_gt_Y_temp
            model_eval_entropy = copy.deepcopy(model_gt_temp).to(DEVICE)
            pick_index1 = []
            for num_trans in range(0,Total_budget,eval_period):
                # print(num_trans)
                loss_gain_list = np.zeros(len(train_loader))
                
                for step,(x,y) in enumerate(train_loader):
                    if step>=C_ini and (step not in pick_index1):
                        # print(step)
                        # dataset_base_X_temp = torch.cat((dataset_base_X, x), 0)
                        # dataset_base_Y_temp = torch.cat((dataset_base_Y, y), 0)
                        
                        dataset_entropy_X_temp =x# torch.cat((dataset_gt_X_temp, x), 0)
                        dataset_entropy_Y_temp =y# torch.cat((dataset_gt_Y_temp, y), 0)
                        
                        # Dataloader_entropy_temp = Data.DataLoader(TensorDataset(dataset_entropy_X_temp,dataset_entropy_Y_temp.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                        
                        y_predict = model_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0]
                        prob_predict = F.softmax(y_predict,dim=1)
                        margin_pred = torch.sort(prob_predict,dim=1)[0]
                        margin_sum = torch.sum(margin_pred[:,-1]- margin_pred[:,-2]).item()
                        # loss =  1/loss_func(model_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                        # loss =  loss_func(model_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                        loss_gain_list[step] = 1/margin_sum
                        
                        
                        # div,_ = estimate_diversity_empirical(model_eval_base,Dataloader_base_temp)
                        # marginal_gain_list[step] = div
                        
                index = np.argsort(loss_gain_list)[-eval_period:]
                # index = np.argsort(loss_gain_list)[0:eval_period]
                pick_index1.extend(index)
                
                for step,(x,y) in enumerate(train_loader):
                    if step in index:
                        dataset_model_entropy_X = torch.cat((dataset_model_entropy_X, x), 0)
                        dataset_model_entropy_Y = torch.cat((dataset_model_entropy_Y, y), 0)
                        # break
                
                if num_trans%eval_period ==0:
                    Dataloader_entropy = Data.DataLoader(TensorDataset(dataset_model_entropy_X,dataset_model_entropy_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
                    model_eval_entropy = EfficientNetB0().to(DEVICE)
                    if multip_GPU:
                        model_eval_entropy = torch.nn.DataParallel(model_eval_entropy)
                    optimizer_entropy = torch.optim.SGD(model_eval_entropy.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_entropy, T_max=200)
                    acc_train_e,acc_test_e = train_cifar(Dataloader_entropy,val_loader,DEVICE,EPOCH,model_eval_entropy,optimizer_entropy,loss_func,scheduler)
                    # acc_train_e,acc_test_e =train_dis(Dataloader_entropy,val_loader,DEVICE,EPOCH,model_eval_entropy,optimizer_entropy,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                    acc_entropy_list[num_trans,0,:]=np.array(acc_train_e)
                    acc_entropy_list[num_trans,1,:]=np.array(acc_test_e)
            print(len(dataset_model_entropy_X))
            Dataloader_entropy = Data.DataLoader(TensorDataset(dataset_model_entropy_X,dataset_model_entropy_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
            model_eval_entropy = EfficientNetB0().to(DEVICE)
            if multip_GPU:
                        model_eval_entropy = torch.nn.DataParallel(model_eval_entropy)
            optimizer_entropy = torch.optim.SGD(model_eval_entropy.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_entropy, T_max=200)
            acc_train_e,acc_test_e = train_cifar(Dataloader_entropy,val_loader,DEVICE,EPOCH,model_eval_entropy,optimizer_entropy,loss_func,scheduler)
            return acc_entropy_list,acc_test_e,pick_index1
    
    
    

        
        ### based marginal rate gain
    def pick_by_rate(eval_period):
        acc_base_list = np.zeros((Total_budget,2,EPOCH))
        dataset_base_X = dataset_gt_X_temp
        dataset_base_Y = dataset_gt_Y_temp
        model_eval_base = copy.deepcopy(model_gt_temp).to(DEVICE)
        pick_index2 = []
        for num_trans in range(0,Total_budget,eval_period):
            # print(num_trans)
            marginal_gain_list = np.zeros(len(train_loader))
            
            for step,(x,y) in enumerate(train_loader):
                if step>=C_ini and (step not in pick_index2):
                    # print(step)
                    # dataset_base_X_temp = torch.cat((dataset_base_X, x), 0)
                    # dataset_base_Y_temp = torch.cat((dataset_base_Y, y), 0)
                    
                    dataset_base_X_temp = torch.cat((dataset_gt_X_temp, x), 0)
                    dataset_base_Y_temp = torch.cat((dataset_gt_Y_temp, y), 0)
                    
                    Dataloader_base_temp = Data.DataLoader(TensorDataset(dataset_base_X_temp,dataset_base_Y_temp.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
                    R,R_c=code_diversity_empirical(Dataloader_base_temp,model_eval_base,num_class=num_class,epsilon= 0.7)
                    marginal_gain_list[step] = R-R_c
                    
                    
                    # div,_ = estimate_diversity_empirical(model_eval_base,Dataloader_base_temp)
                    # marginal_gain_list[step] = div
                    
            index = np.argsort(marginal_gain_list)[-eval_period:]
            pick_index2.extend(index)
            
            for step,(x,y) in enumerate(train_loader):
                if step in index:
                    dataset_base_X = torch.cat((dataset_base_X, x), 0)
                    dataset_base_Y = torch.cat((dataset_base_Y, y), 0)
                    # break
            
            if num_trans%eval_period ==0:
                Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
                model_eval_base = EfficientNetB0().to(DEVICE)
                if multip_GPU:
                        model_eval_base = torch.nn.DataParallel(model_eval_base)
                optimizer_eval = torch.optim.SGD(model_eval_base.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_eval, T_max=200)
                acc_train_,acc_test_ = train_cifar(Dataloader_base,val_loader,DEVICE,EPOCH,model_eval_base,optimizer_eval,loss_func,scheduler)
                # acc_train_,acc_test_ =train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,model_eval_base,optimizer_eval,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                acc_base_list[num_trans,0,:]=np.array(acc_train_)
                acc_base_list[num_trans,1,:]=np.array(acc_test_)
        
        Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
        model_base = EfficientNetB0().to(DEVICE)
        if multip_GPU:
                        model_base = torch.nn.DataParallel(model_base)
        optimizer_base = torch.optim.SGD(model_base.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, T_max=200)
        acc_train_base,acc_test_base = train_cifar(Dataloader_base,val_loader,DEVICE,EPOCH,model_base,optimizer_base,loss_func,scheduler)
        return acc_base_list,acc_test_base,pick_index2
    # acc_train_base,acc_test_base = train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,model_base,optimizer_base,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
    
    # acc_train_gt,acc_test_gt = train_cifar(train_loader,val_loader,DEVICE,EPOCH,model_gt,optimizer_gt,loss_func,scheduler)
    acc_entropy_list_f_all = []
    acc_base_list_f_all = []
    
    acc_entropy_list_all = []
    acc_base_list_all = []
    
    
    
    
    def RD_DPP_diveristy(eval_period):
        acc_base_list = np.zeros((Total_budget,2,EPOCH))
        dataset_base_X = dataset_gt_X_temp
        dataset_base_Y = dataset_gt_Y_temp
        model_eval_base = copy.deepcopy(model_gt_temp).to(DEVICE)
        pick_index2 = []
        Div_count = []
        for num_trans in range(0,Total_budget,eval_period):
            # print(num_trans)
            marginal_gain_list = np.zeros(len(train_loader))
            feats_all = np.zeros((len(train_loader),num_class*num_feat))+1e-8# len(feats)*num_class
            step_exlude =[]
            Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
            R_0,R_c0 =code_diversity_empirical_old(Dataloader_base,model_eval_base,num_class=num_class,epsilon= 0.7)
            Div_count.append(R_0-R_c0)
            for step,(x,y) in enumerate(train_loader):
                if step>=C_ini and (step not in pick_index2) and len(x)==BATCH_SIZE:
                    # print(step)
                    dataset_base_X_temp = torch.cat((dataset_base_X, x), 0)
                    dataset_base_Y_temp = torch.cat((dataset_base_Y, y), 0)
                    
                    # dataset_base_X_temp = torch.cat((dataset_gt_X_temp, x), 0)
                    # dataset_base_Y_temp = torch.cat((dataset_gt_Y_temp, y), 0)
                    
                    Dataloader_base_temp = Data.DataLoader(TensorDataset(dataset_base_X_temp,dataset_base_Y_temp.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
                    
                    
                    
                    
                    R,R_c=code_diversity_empirical_old(Dataloader_base_temp,model_eval_base,num_class=num_class,epsilon= 0.7)
                    marginal_gain_list[step] =R-R_c #R-R_c #np.log(R-R_c)
                    Dataloader_base_temp2 = Data.DataLoader(TensorDataset(x,y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
                    feats_all[step] = feat_empirical(Dataloader_base_temp2,model_eval_base,num_class=num_class,feat_len = num_feat)
                
                elif len(x)<BATCH_SIZE:
                    step_exlude.append(step)
                else:
                    step_exlude.append(step)
                    
                    # div,_ = estimate_diversity_empirical(model_eval_base,Dataloader_base_temp)
                    # marginal_gain_list[step] = div
                    
            # feats_all=feats_all[C_ini:]
            feats_all /= np.linalg.norm(feats_all, axis=1, keepdims=True)
            similarity_m =  feats_all@ feats_all.T      
            # similarity_m = (1+similarity_m)/2
            # marginal_gain_list[marginal_gain_list>0]
            # marginal_gain_list[marginal_gain_list>0] = (marginal_gain_list[marginal_gain_list>0]- \
                                            # np.min(marginal_gain_list[marginal_gain_list>0]))/(np.max(marginal_gain_list[marginal_gain_list>0])-np.min(marginal_gain_list[marginal_gain_list>0]))
            marginal_gain_list = (marginal_gain_list-np.min(marginal_gain_list))/(np.max(marginal_gain_list)-np.min(marginal_gain_list))
            
            kernel =  marginal_gain_list.reshape((-1, 1)) * similarity_m * marginal_gain_list.reshape((1, -1))
            # kernel = similarity_m
            # kernel = np.exp(kernel)
            
            for i in step_exlude:
                kernel[i,:]=0
                kernel[:,i]=0
            
            # kernel = (kernel-np.min(kernel))/(np.max(kernel)-np.min(kernel))
            index = dpp(kernel, max_length=eval_period, epsilon=1E-20)
            
            
            print(index)
            print(marginal_gain_list[index])
            
            
            
            while len(set(index))<eval_period:
                for  i in index:
                    kernel[i,:]=0
                    kernel[:,i]=0
                
                index_new = dpp(kernel, max_length=eval_period-len(set(index)), epsilon=1E-20)
                index.extend(index_new)
                print(index)
                
                
            index_f = index # [i+C_ini for i in index]
            pick_index2.extend(index_f)
            
            for step,(x,y) in enumerate(train_loader):
                if step in index_f:
                    print(step)
                    dataset_base_X = torch.cat((dataset_base_X, x), 0)
                    dataset_base_Y = torch.cat((dataset_base_Y, y), 0)
                    # break
            
            if num_trans%eval_period ==0:
                print('DPP')
                Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
                model_eval_base = EfficientNetB0().to(DEVICE)
                if multip_GPU:
                        model_eval_base = torch.nn.DataParallel(model_eval_base)
                optimizer_eval = torch.optim.SGD(model_eval_base.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_eval, T_max=200)
                acc_train_,acc_test_ = train_cifar(Dataloader_base,val_loader,DEVICE,EPOCH,model_eval_base,optimizer_eval,loss_func,scheduler)
                # acc_train_,acc_test_ =train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,model_eval_base,optimizer_eval,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                acc_base_list[num_trans,0,:]=np.array(acc_train_)
                acc_base_list[num_trans,1,:]=np.array(acc_test_)
                print(np.mean(acc_test_[-20:]))
        Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
        model_base = EfficientNetB0().to(DEVICE)
        if multip_GPU:
                        model_base = torch.nn.DataParallel(model_base)
        optimizer_base = torch.optim.SGD(model_base.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, T_max=200)
        acc_train_base,acc_test_base = train_cifar(Dataloader_base,val_loader,DEVICE,EPOCH,model_base,optimizer_base,loss_func,scheduler)
        print("the length of dpp is ")
        print(len(dataset_base_X))
        
        np.save(path+f'div_{run_index}.npy',np.array(Div_count))
        
        return acc_base_list,acc_test_base,pick_index2
    
    
    
    def pick_by_adptive(eval_period,bordline,uncertain='minmargin'):
        acc_base_list = np.zeros((Total_budget,2,EPOCH))
        dataset_base_X = dataset_gt_X_temp
        dataset_base_Y = dataset_gt_Y_temp
        model_eval_base = copy.deepcopy(model_gt_temp).to(DEVICE)
        pick_index2 = []
        Div_count = []
        
        Status_list = []
        
        Flag = True
        for num_trans in range(0,Total_budget,eval_period):
            print(Flag)
            marginal_gain_list = np.zeros(len(train_loader))
            feats_all = np.zeros((len(train_loader),num_class*num_feat))+1e-8# len(feats)*num_class
            step_exlude =[]
            Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
            R_0,R_c0 =code_diversity_empirical_old(Dataloader_base,model_eval_base,num_class=num_class,epsilon= 0.7)
            Div_count.append(R_0-R_c0)
            if (num_trans>0 and Div_count[-1]-Div_count[-2]>bordline and Flag == True) or num_trans==0:
                print('DPP select')
                Status_list.append(1)
                for step,(x,y) in enumerate(train_loader):
                    if step>=C_ini and (step not in pick_index2) and len(x)==BATCH_SIZE:
                        # print(step)
                        dataset_base_X_temp = torch.cat((dataset_base_X, x), 0)
                        dataset_base_Y_temp = torch.cat((dataset_base_Y, y), 0)
                        
                        # dataset_base_X_temp = torch.cat((dataset_gt_X_temp, x), 0)
                        # dataset_base_Y_temp = torch.cat((dataset_gt_Y_temp, y), 0)
                        
                        Dataloader_base_temp = Data.DataLoader(TensorDataset(dataset_base_X_temp,dataset_base_Y_temp.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
                        
                        
                        
                        
                        R,R_c=code_diversity_empirical_old(Dataloader_base_temp,model_eval_base,num_class=num_class,epsilon= 0.7)
                        marginal_gain_list[step] =R-R_c #R-R_c #np.log(R-R_c)
                        Dataloader_base_temp2 = Data.DataLoader(TensorDataset(x,y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
                        feats_all[step] = feat_empirical(Dataloader_base_temp2,model_eval_base,num_class=num_class,feat_len = num_feat)
                    
                    elif len(x)<BATCH_SIZE:
                        step_exlude.append(step)
                    else:
                        step_exlude.append(step)
                        
                        # div,_ = estimate_diversity_empirical(model_eval_base,Dataloader_base_temp)
                        # marginal_gain_list[step] = div
                        
                # feats_all=feats_all[C_ini:]
                feats_all /= np.linalg.norm(feats_all, axis=1, keepdims=True)
                similarity_m =  feats_all@ feats_all.T      
                # similarity_m = (1+similarity_m)/2
                # marginal_gain_list[marginal_gain_list>0]
                # marginal_gain_list[marginal_gain_list>0] = (marginal_gain_list[marginal_gain_list>0]- \
                                                # np.min(marginal_gain_list[marginal_gain_list>0]))/(np.max(marginal_gain_list[marginal_gain_list>0])-np.min(marginal_gain_list[marginal_gain_list>0]))
                marginal_gain_list = (marginal_gain_list-np.min(marginal_gain_list))/(np.max(marginal_gain_list)-np.min(marginal_gain_list))
                
                kernel =  marginal_gain_list.reshape((-1, 1)) * similarity_m * marginal_gain_list.reshape((1, -1))
                # kernel = similarity_m
                # kernel = np.exp(kernel)
                
                for i in step_exlude:
                    kernel[i,:]=0
                    kernel[:,i]=0
                
                # kernel = (kernel-np.min(kernel))/(np.max(kernel)-np.min(kernel))
                index = dpp(kernel, max_length=eval_period, epsilon=1E-20)
                
                
                # print(index)
                # print(marginal_gain_list[index])
                
                
                
                while len(set(index))<eval_period:
                    for  i in index:
                        kernel[i,:]=0
                        kernel[:,i]=0
                    
                    index_new = dpp(kernel, max_length=eval_period-len(set(index)), epsilon=1E-20)
                    index.extend(index_new)
                    print(index)
                    
                    
                index_f = index # [i+C_ini for i in index]
                pick_index2.extend(index_f)
            # elif num_trans>0 and Div_count[-1]-Div_count[-2]<=bordline or Flag == False:
            else:
                Flag = False
                Status_list.append(0)
                print('margin select')
                loss_gain_list=  np.zeros(len(train_loader))
                for step,(x,y) in enumerate(train_loader):
                    if step>=C_ini and (step not in pick_index2) and len(x)==BATCH_SIZE:    
                        y_predict = model_eval_base(x.to(DEVICE))[0]
                        prob_predict = F.softmax(y_predict,dim=1)
                        margin_pred = torch.sort(prob_predict,dim=1)[0]
                        margin_sum = torch.sum(margin_pred[:,-1]- margin_pred[:,-2]).item()
                        # loss =  1/loss_func(model_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                        loss =  loss_func(y_predict,y.to(DEVICE)).item()
                        if uncertain=='minmargin':
                            loss_gain_list[step] = 1/margin_sum
                        else:
                            loss_gain_list[step] = loss
                        
                        
                        
                        # div,_ = estimate_diversity_empirical(model_eval_base,Dataloader_base_temp)
                        # marginal_gain_list[step] = div
                        
                index_f = np.argsort(loss_gain_list)[-eval_period:]
                    
                    
                    
                    
                pick_index2.extend(index_f)
            for step,(x,y) in enumerate(train_loader):
                if step in index_f:
                    print(step)
                    dataset_base_X = torch.cat((dataset_base_X, x), 0)
                    dataset_base_Y = torch.cat((dataset_base_Y, y), 0)
                    # break
            
            
                
                    
                    
            
            if num_trans%eval_period ==0:
                # print('DPP')
                Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True) #BATCH_SIZE
                model_eval_base = EfficientNetB0().to(DEVICE)
                if multip_GPU:
                        model_eval_base = torch.nn.DataParallel(model_eval_base)
                optimizer_eval = torch.optim.SGD(model_eval_base.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_eval, T_max=200)
                acc_train_,acc_test_ = train_cifar(Dataloader_base,val_loader,DEVICE,EPOCH,model_eval_base,optimizer_eval,loss_func,scheduler)
                # acc_train_,acc_test_ =train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,model_eval_base,optimizer_eval,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                acc_base_list[num_trans,0,:]=np.array(acc_train_)
                acc_base_list[num_trans,1,:]=np.array(acc_test_)
                print(np.mean(acc_test_[-20:]))
        Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=train_batch,shuffle=True,pin_memory=True)
        model_base = EfficientNetB0().to(DEVICE)
        if multip_GPU:
                        model_base = torch.nn.DataParallel(model_base)
        optimizer_base = torch.optim.SGD(model_base.parameters(), lr=args.LR,
                      momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, T_max=200)
        acc_train_base,acc_test_base = train_cifar(Dataloader_base,val_loader,DEVICE,EPOCH,model_base,optimizer_base,loss_func,scheduler)
        print("the length of ada is ")
        print(len(dataset_base_X))
        return acc_base_list,acc_test_base,Status_list,pick_index2
        # acc_train_base,acc_test_base = train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,model_base,optimizer_base,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
        
        # acc_train_gt,acc_test_gt = train_cifar(train_loader,val_loader,DEVICE,EPOCH,model_gt,optimizer_gt,loss_func,scheduler)
    acc_entropy_list_f_all = []
    acc_margin_list_f_all = []
    acc_base_list_f_all = []
    acc_base2_list_f_all = []
    acc_dpp_list_f_all = []
    acc_dpp_list_f_old_all = []
    acc_adpative_1_f_all = []
    acc_adpative_2_f_all = []
    
    acc_entropy_list_all = []
    acc_margin_list_all = []
    acc_base_list_all = []
    acc_dpp_list_all = []
    acc_base2_list_all = []
    acc_dpp_old_list_all = []
    acc_adpative_1_list_all = []
    acc_adpative_2_list_all = []
    # plt.figure()
    # plt.plot(acc_test_base)
    # plt.plot(acc_test_dpp)
    
    
    
    
    for eval_period in eval_period_list:
        
        print('Entropy')
        acc_entropy_list,acc_test_e,e_index = pick_by_entropy(eval_period)
        print('Min margin')
        acc_margin_list,acc_test_margin,margin_index = pick_by_min_margin(eval_period)
        
        print('margin rate gain')
        acc_base_list,acc_test_base,base_index =pick_by_rate(eval_period)
        print('RD DPP')
        acc_dpp_list_old,acc_test_dpp_old,dpp_index =RD_DPP_diveristy(eval_period)


        acc_a1_list,acc_test_a1,Status_list1,a1_index = [],[],[],[]
        acc_a2_list,acc_test_a2,Status_list2,a2_index = [],[],[],[]
        
        
    acc_rand_list,acc_test_rand_all,rand_index = pick_by_random(5)
    
    train_loader= Data.DataLoader(TensorDataset(dataset_pre_X,dataset_pre_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=False,pin_memory=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gt, T_max=200)
    acc_train_gt,acc_test_gt = train_cifar(train_loader,val_loader,DEVICE,EPOCH,model_gt,optimizer_gt,loss_func,scheduler)
    
    
    
    result = [acc_rand_list, acc_entropy_list, acc_margin_list, acc_base_list, acc_dpp_list_old,acc_a1_list,acc_a2_list, Status_list1, Status_list2, e_index,margin_index,base_index,dpp_index,a1_index,a2_index ]
    results = np.array(result,dtype=object)
    
    
    name =  f'{args.dataset_name}_{run_index}'
    
    np.save(path+name+'.npy', results)


plt.figure()
plt.plot(np.mean(acc_rand_list[::args.T,1,-20:],axis = 1))
plt.plot(np.mean(acc_entropy_list[::args.T,1,-20:],axis = 1))
plt.plot(np.mean(acc_margin_list[::args.T,1,-20:],axis = 1))
plt.plot(np.mean(acc_base_list[::args.T,1,-20:],axis = 1)  )
plt.plot(np.mean(acc_dpp_list_old[::args.T,1,-20:],axis = 1))
plt.legend(['rand','entropy','min-margin','margin gain','rd-dpp'])




plt.figure()
plt.plot(Status_list1)
plt.plot(Status_list2)

