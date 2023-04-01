# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:50:28 2022

@author: xiwenc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:08:07 2022

@author: xiwenc
"""








"""
Created on Thu Oct 20 21:10:01 2022

@author: xiwenc
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 21:29:45 2022

@author: Xiwen Chen

No noise

"""


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:32'


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
import copy
import random
from functions_emp import train,test, \
code_diversity_empirical,code_diversity_empirical_old,feat_empirical,dpp,LogisticRegression,CNN,image_gau_noise

from torch.utils.data import TensorDataset

import sys
# from data_split import mnist_noniid,DatasetSplit
import json
# import pynvml
# import psutil
# import matlab
# import matlab.engine
# eng = matlab.engine.start_matlab()



import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--EPOCH', type=int, default=100, help="epoches of training")
    parser.add_argument('--index', type=int, default=78, help="index of filename")
    parser.add_argument('--var', type=float, default=0.5, help="variance of noise")
    parser.add_argument('--BATCH_SIZE', type=int, default=64, help="batch size")
    parser.add_argument('--LR', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--Total_budget', type=int, default=60, help="Total budget")
    parser.add_argument('--T_BS', type=int, default=128, help="training batch size")
    parser.add_argument('--clean_rate', type=float, default=5/50, help="clean data rate")
    parser.add_argument('--Ns_rate', type=float, default=0.2, help="rate of the received data for Fisher Information Hist")
    parser.add_argument('--fisher_ratio', type=float, default=0.8, help="param for threshold 1")
    # model arguments
    parser.add_argument('--num_mask', type=int, default=4, help='number of mask')
    parser.add_argument('--max_re', type=int, default=5, help='max number of retrans for one batch')
    parser.add_argument('--quantile_ratio', type=float, default=0.5, help="clean data rate")
    # other arguments
    parser.add_argument('--dataset_name', type=str, default='MNIST', help="name of dataset")
    parser.add_argument('--name_index', type=int, default=1, help="name index")
    parser.add_argument('--repeat', type=int, default=5, help="number of repeat")
    #for channel coding scheme
    
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse args
    args = args_parser()
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

   

    epsilon = np.sqrt(0.5)
    EPOCH =  args.EPOCH
    BATCH_SIZE= args.BATCH_SIZE
    BS_train =4#  BATCH_SIZE
    LR = args.LR
    Total_budget = args.Total_budget
    fisher_ratio = args.fisher_ratio
    
    C_ini = 5 #int(Total_budget*args.clean_rate)
    
    C_ini_2 = int(Total_budget*args.Ns_rate)
    
    C_ini_2_2 =C_ini+0
    
    
    eval_period = 5 #Total_budget #  Total_budget
    Re_total = args.max_re
    Fisher_total = 0
    num_mask = args.num_mask
    
    eval_period_list = [5,10]
    
    dataset_name = args.dataset_name
    
    #filename = '{}_Total_{}_ratio_{}_numMask_{}_'.format(args.name_index,Total_budget,fisher_ratio,num_mask)+dataset_name
    
    
    
    
    
    path = 'exp2_MNIST/'
    os.makedirs(path,exist_ok=True)
    
    
    # task_name = 'MNIST'
    
    DOWNLOAD_MINIST = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    print(f'use device: {DEVICE}')
    
    
    # get_avaliable_memory(DEVICE)
    train_data = torchvision.datasets.MNIST(
        root = 'data/MNIST',  #
        train = True,       #False: tEST
    #     transform = torchvision.transforms.ToTensor(),
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),#norm
        download=DOWNLOAD_MINIST
    )
    
    # train_data = torchvision.datasets.FashionMNIST(
    #     root = './FashionMNIST',  #
    #     train = True,       #False: tEST
    # #     transform = torchvision.transforms.ToTensor(),
    #     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),#norm
    #     download=DOWNLOAD_MINIST
    # )
    
    
    print(train_data.data.size())
    print(train_data.targets.size())
    
    
    # load train data and test data
    # train_loader_gt = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
    
    
    
    # dict_users = mnist_noniid(train_data, BATCH_SIZE)
    for run_index in range(args.repeat):
        filename = '{}_Total_{}_batch_{}'.format(run_index,Total_budget,args.BATCH_SIZE)+dataset_name
        with open('data/MNIST.txt') as f:
            data = f.read()
            
        dict_users = json.loads(data)
        
        all_len =len(dict_users)
        
        
    # -*- coding: utf-8 -*-
        
        Noise_array = np.zeros((all_len,10),dtype=object)
        num_mask = args.num_mask
        
        # for i in range(len(train_loader_gt)):
        #     for j in range(10):
        #         index_list= list(np.arange(16))
        #         num_mask = 0#2#np.random.randint(0,6)   #np.random.randint(args.num_mask-2,args.num_mask+3)
        #         Noise_array[i,j]=random.sample(index_list,num_mask)
        # test_data = torchvision.datasets.MNIST(root='./MNIST',train=False)
        
        
        # Noise_array = np.random.rand(all_len,10)#*2
        
        test_data = torchvision.datasets.MNIST(root='data/MNIST',train=False)
        
        dataset_pre_X = torch.tensor([])
        dataset_pre_Y = torch.tensor([])
        
       
        dataset_gt_X_temp = torch.tensor([])
        dataset_gt_Y_temp = torch.tensor([])
        
        
        dataset_pre_X_all = torch.tensor([])
        dataset_pre_Y_all = torch.tensor([])
        
        
        all_index_cout = 0
        init_index = []
        for step in range(all_len):
            idxs = dict_users[str(step)]
            rest = len(idxs)%BATCH_SIZE
            idxs = idxs[:len(idxs)-rest]
            x= train_data.data[idxs]
            y =train_data.targets[idxs]
            x = x.unsqueeze(1)
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
                
            
        
        train_loader_all= Data.DataLoader(TensorDataset(dataset_pre_X_all,dataset_pre_Y_all.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=False,pin_memory=True)
            
        
        
        
        
        
        
        train_loader= Data.DataLoader(TensorDataset(dataset_pre_X,dataset_pre_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=False,pin_memory=True)
            
        #generate test data
        test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
        test_y = test_data.targets[:2000]
        val_loader = Data.DataLoader(dataset=TensorDataset(test_x,test_y),batch_size=BATCH_SIZE,shuffle=True)
        
        print(dataset_pre_X.size())
        print(dataset_pre_Y.size())
        
        cnn_gt = CNN(28,28,1,num_hidden=2).to(DEVICE)
        cnn_gt_temp =copy.deepcopy(cnn_gt).to(DEVICE)
        # cnn_gt_part= CNN(28,28,1,num_hidden=2).to(DEVICE)
        cnn_noise =copy.deepcopy(cnn_gt).to(DEVICE)
        cnn_base= copy.deepcopy(cnn_gt).to(DEVICE)
        cnn_entropy=copy.deepcopy(cnn_gt).to(DEVICE)
        cnn_rand = copy.deepcopy(cnn_gt).to(DEVICE)
        LR0 =1e-4
        optimizer_gt = torch.optim.Adam(cnn_gt.parameters(),lr=LR)
        # optimizer_gt_part = torch.optim.Adam(cnn_gt_part.parameters(),lr=LR)
        optimizer_gt_temp = torch.optim.Adam(cnn_gt_temp.parameters(),lr=LR0)
        optimizer_noise = torch.optim.Adam(cnn_noise.parameters(),lr=LR)
        optimizer_base = torch.optim.Adam(cnn_base.parameters(),lr=LR)
        optimizer_entropy= torch.optim.Adam(cnn_entropy.parameters(),lr=LR)
        optimizer_rand= torch.optim.Adam(cnn_rand.parameters(),lr=LR)
        # optimizer_base_2 = torch.optim.Adam(cnn_base_2.parameters(),lr=LR)
        
        loss_func = nn.CrossEntropyLoss()
        
        Dataloader_gt_temp= Data.DataLoader(TensorDataset(dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor)),batch_size=BS_train,shuffle=True,pin_memory=True)
        acc_train_gt_temp,acc_test_gt_temp = train(Dataloader_gt_temp,val_loader,DEVICE,EPOCH,cnn_gt_temp,optimizer_gt_temp,loss_func)
        cnn_gt_temp.eval()
        
        cnn_eval_base = copy.deepcopy(cnn_gt_temp).to(DEVICE)
        cnn_eval_gt = copy.deepcopy(cnn_gt_temp).to(DEVICE)
        cnn_eval_entropy = copy.deepcopy(cnn_gt_temp).to(DEVICE)
        
        
        
        
        
        
        


        
        
        ### find valued batch
        # rand pick
        # count = 0
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
                    cnn_rand = CNN(28,28,1,num_hidden=2).to(DEVICE)
                    optimizer_rand= torch.optim.Adam(cnn_rand.parameters(),lr=LR)
                    acc_train_rand,acc_test_rand = train(Dataloader_rand,val_loader,DEVICE,EPOCH,cnn_rand,optimizer_rand,loss_func)
                    
                    acc_rand_list[num_trans,0,:]=np.array(acc_train_rand)
                    acc_rand_list[num_trans,1,:]=np.array(acc_test_rand)
                    
            Dataloader_rand = Data.DataLoader(TensorDataset(dataset_rand_X,dataset_rand_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
            print(len(dataset_rand_X))
            cnn_rand = CNN(28,28,1,num_hidden=2).to(DEVICE)
            optimizer_rand= torch.optim.Adam(cnn_rand.parameters(),lr=LR)
            acc_train_rand,acc_test_rand = train(Dataloader_rand,val_loader,DEVICE,EPOCH,cnn_rand,optimizer_rand,loss_func)
            return acc_rand_list,acc_test_rand
        
        
        # set(list(range(C_ini,len( train_loader))))-
       
        
        def pick_by_random_all(eval_period):
            acc_rand_list = np.zeros((Total_budget,2,EPOCH))
            dataset_rand_X = dataset_gt_X_temp
            dataset_rand_Y = dataset_gt_Y_temp
            # eval_period =10
            # rand_index_list = random.sample(list(range(C_ini,len( train_loader))),Total_budget)
            rand_index_list = random.sample(set(range(len( train_loader_all)))-set(init_index),Total_budget)
            for num_trans in range(0,Total_budget,eval_period):
                rand_index = rand_index_list[num_trans:num_trans+eval_period]
                print(rand_index)
                for step,(x,y) in enumerate(train_loader_all):
                    
                    if step in  rand_index:
                        # count = count+1
                        dataset_rand_X = torch.cat((dataset_rand_X, x), 0)
                        dataset_rand_Y = torch.cat((dataset_rand_Y, y), 0)
                
                if num_trans%eval_period ==0:
                    Dataloader_rand = Data.DataLoader(TensorDataset(dataset_rand_X,dataset_rand_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
                    cnn_rand = CNN(28,28,1,num_hidden=2).to(DEVICE)
                    optimizer_rand= torch.optim.Adam(cnn_rand.parameters(),lr=LR)
                    acc_train_rand,acc_test_rand = train(Dataloader_rand,val_loader,DEVICE,EPOCH,cnn_rand,optimizer_rand,loss_func)
                    
                    acc_rand_list[num_trans,0,:]=np.array(acc_train_rand)
                    acc_rand_list[num_trans,1,:]=np.array(acc_test_rand)
                    
            Dataloader_rand = Data.DataLoader(TensorDataset(dataset_rand_X,dataset_rand_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
            print(len(dataset_rand_X))
            cnn_rand = CNN(28,28,1,num_hidden=2).to(DEVICE)
            optimizer_rand= torch.optim.Adam(cnn_rand.parameters(),lr=LR)
            acc_train_rand,acc_test_rand = train(Dataloader_rand,val_loader,DEVICE,EPOCH,cnn_rand,optimizer_rand,loss_func)
            return acc_rand_list,acc_test_rand
        # plt.imshow(x[0,0],'gray')
        ### based on accuracy
        # cnn_eval_base = copy.deepcopy(cnn_gt_temp).to(DEVICE)
        # cnn_eval_gt = copy.deepcopy(cnn_gt_temp).to(DEVICE)
        # cnn_eval_entropy = copy.deepcopy(cnn_gt_temp).to(DEVICE)
        
        
        
        
        
        
        def pick_by_entropy(eval_period):  #uncertainty
            acc_entropy_list = np.zeros((Total_budget,2,EPOCH))
            dataset_model_entropy_X= dataset_gt_X_temp
            dataset_model_entropy_Y = dataset_gt_Y_temp
            cnn_eval_entropy = copy.deepcopy(cnn_gt_temp).to(DEVICE)
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
                        
                        # loss =  1/loss_func(cnn_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                        loss =  loss_func(cnn_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                        loss_gain_list[step] = loss
                        
                        
                        # div,_ = estimate_diversity_empirical(cnn_eval_base,Dataloader_base_temp)
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
                    cnn_eval_entropy = CNN(28,28,1,num_hidden=2).to(DEVICE)
                    optimizer_entropy = torch.optim.Adam(cnn_eval_entropy.parameters(),lr=LR)
                    acc_train_e,acc_test_e = train(Dataloader_entropy,val_loader,DEVICE,EPOCH,cnn_eval_entropy,optimizer_entropy,loss_func)
                    # acc_train_e,acc_test_e =train_dis(Dataloader_entropy,val_loader,DEVICE,EPOCH,cnn_eval_entropy,optimizer_entropy,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                    acc_entropy_list[num_trans,0,:]=np.array(acc_train_e)
                    acc_entropy_list[num_trans,1,:]=np.array(acc_test_e)
            Dataloader_entropy = Data.DataLoader(TensorDataset(dataset_model_entropy_X,dataset_model_entropy_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
            cnn_eval_entropy = CNN(28,28,1,num_hidden=2).to(DEVICE)
            optimizer_entropy = torch.optim.Adam(cnn_eval_entropy.parameters(),lr=LR)
            acc_train_e,acc_test_e = train(Dataloader_entropy,val_loader,DEVICE,EPOCH,cnn_eval_entropy,optimizer_entropy,loss_func)
            return acc_entropy_list,acc_test_e
        
        def pick_by_min_margin(eval_period):  #uncertainty
            acc_entropy_list = np.zeros((Total_budget,2,EPOCH))
            dataset_model_entropy_X= dataset_gt_X_temp
            dataset_model_entropy_Y = dataset_gt_Y_temp
            cnn_eval_entropy = copy.deepcopy(cnn_gt_temp).to(DEVICE)
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
                        
                        y_predict = cnn_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0]
                        prob_predict = F.softmax(y_predict,dim=1)
                        margin_pred = torch.sort(prob_predict,dim=1)[0]
                        margin_sum = torch.sum(margin_pred[:,-1]- margin_pred[:,-2]).item()
                        # loss =  1/loss_func(cnn_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                        # loss =  loss_func(cnn_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                        loss_gain_list[step] = 1/margin_sum
                        
                        
                        # div,_ = estimate_diversity_empirical(cnn_eval_base,Dataloader_base_temp)
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
                    cnn_eval_entropy = CNN(28,28,1,num_hidden=2).to(DEVICE)
                    optimizer_entropy = torch.optim.Adam(cnn_eval_entropy.parameters(),lr=LR)
                    acc_train_e,acc_test_e = train(Dataloader_entropy,val_loader,DEVICE,EPOCH,cnn_eval_entropy,optimizer_entropy,loss_func)
                    # acc_train_e,acc_test_e =train_dis(Dataloader_entropy,val_loader,DEVICE,EPOCH,cnn_eval_entropy,optimizer_entropy,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                    acc_entropy_list[num_trans,0,:]=np.array(acc_train_e)
                    acc_entropy_list[num_trans,1,:]=np.array(acc_test_e)
            print(len(dataset_model_entropy_X))
            Dataloader_entropy = Data.DataLoader(TensorDataset(dataset_model_entropy_X,dataset_model_entropy_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True)
            cnn_eval_entropy = CNN(28,28,1,num_hidden=2).to(DEVICE)
            optimizer_entropy = torch.optim.Adam(cnn_eval_entropy.parameters(),lr=LR)
            acc_train_e,acc_test_e = train(Dataloader_entropy,val_loader,DEVICE,EPOCH,cnn_eval_entropy,optimizer_entropy,loss_func)
            return acc_entropy_list,acc_test_e
        
        def pick_by_rate2(eval_period):
            acc_base_list = np.zeros((Total_budget,2,EPOCH))
            dataset_base_X = dataset_gt_X_temp
            dataset_base_Y = dataset_gt_Y_temp
            cnn_eval_base = copy.deepcopy(cnn_gt_temp).to(DEVICE)
            pick_index2 = []
            for num_trans in range(0,Total_budget,eval_period):
                # print(num_trans)
                marginal_gain_list = np.zeros(len(train_loader))
                
                for step,(x,y) in enumerate(train_loader):
                    if step>=C_ini and (step not in pick_index2):
                        # print(step)
                        dataset_base_X_temp = torch.cat((dataset_base_X, x), 0)
                        dataset_base_Y_temp = torch.cat((dataset_base_Y, y), 0)
                        
                        # dataset_base_X_temp = torch.cat((dataset_gt_X_temp, x), 0)
                        # dataset_base_Y_temp = torch.cat((dataset_gt_Y_temp, y), 0)
                        
                        Dataloader_base_temp = Data.DataLoader(TensorDataset(dataset_base_X_temp,dataset_base_Y_temp.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                        R,R_c=code_diversity_empirical(Dataloader_base_temp,cnn_eval_base,num_class=10,epsilon= 0.7)
                        marginal_gain_list[step] = R-R_c
                        
                        
                        # div,_ = estimate_diversity_empirical(cnn_eval_base,Dataloader_base_temp)
                        # marginal_gain_list[step] = div
                        
                index = np.argsort(marginal_gain_list)[-eval_period:]
                pick_index2.extend(index)
                
                for step,(x,y) in enumerate(train_loader):
                    if step in index:
                        dataset_base_X = torch.cat((dataset_base_X, x), 0)
                        dataset_base_Y = torch.cat((dataset_base_Y, y), 0)
                        # break
                
                if num_trans%eval_period ==0:
                    Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                    cnn_eval_base = CNN(28,28,1,num_hidden=2).to(DEVICE)
                    optimizer_eval = torch.optim.Adam(cnn_eval_base.parameters(),lr=LR)
                    acc_train_,acc_test_ = train(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_eval_base,optimizer_eval,loss_func)
                    # acc_train_,acc_test_ =train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_eval_base,optimizer_eval,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                    acc_base_list[num_trans,0,:]=np.array(acc_train_)
                    acc_base_list[num_trans,1,:]=np.array(acc_test_)
            
            Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
            cnn_base = CNN(28,28,1,num_hidden=2).to(DEVICE)
            optimizer_base = torch.optim.Adam(cnn_base.parameters(),lr=LR)
            acc_train_base,acc_test_base = train(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_base,optimizer_base,loss_func)
            return acc_base_list,acc_test_base
        # acc_train_base,acc_test_base = train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_base,optimizer_base,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
        
        # acc_train_gt,acc_test_gt = train(train_loader,val_loader,DEVICE,EPOCH,cnn_gt,optimizer_gt,loss_func)
        
        
        ### based marginal rate gain
        def pick_by_rate(eval_period):
            acc_base_list = np.zeros((Total_budget,2,EPOCH))
            dataset_base_X = dataset_gt_X_temp
            dataset_base_Y = dataset_gt_Y_temp
            cnn_eval_base = copy.deepcopy(cnn_gt_temp).to(DEVICE)
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
                        
                        Dataloader_base_temp = Data.DataLoader(TensorDataset(dataset_base_X_temp,dataset_base_Y_temp.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                        R,R_c=code_diversity_empirical(Dataloader_base_temp,cnn_eval_base,num_class=10,epsilon= 0.7)
                        marginal_gain_list[step] = R-R_c
                        
                        
                        # div,_ = estimate_diversity_empirical(cnn_eval_base,Dataloader_base_temp)
                        # marginal_gain_list[step] = div
                        
                index = np.argsort(marginal_gain_list)[-eval_period:]
                pick_index2.extend(index)
                
                for step,(x,y) in enumerate(train_loader):
                    if step in index:
                        dataset_base_X = torch.cat((dataset_base_X, x), 0)
                        dataset_base_Y = torch.cat((dataset_base_Y, y), 0)
                        # break
                
                if num_trans%eval_period ==0:
                    Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                    cnn_eval_base = CNN(28,28,1,num_hidden=2).to(DEVICE)
                    optimizer_eval = torch.optim.Adam(cnn_eval_base.parameters(),lr=LR)
                    acc_train_,acc_test_ = train(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_eval_base,optimizer_eval,loss_func)
                    # acc_train_,acc_test_ =train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_eval_base,optimizer_eval,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                    acc_base_list[num_trans,0,:]=np.array(acc_train_)
                    acc_base_list[num_trans,1,:]=np.array(acc_test_)
            
            Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
            cnn_base = CNN(28,28,1,num_hidden=2).to(DEVICE)
            optimizer_base = torch.optim.Adam(cnn_base.parameters(),lr=LR)
            acc_train_base,acc_test_base = train(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_base,optimizer_base,loss_func)
            return acc_base_list,acc_test_base
        # acc_train_base,acc_test_base = train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_base,optimizer_base,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
        
        # acc_train_gt,acc_test_gt = train(train_loader,val_loader,DEVICE,EPOCH,cnn_gt,optimizer_gt,loss_func)
        acc_entropy_list_f_all = []
        acc_base_list_f_all = []
        
        acc_entropy_list_all = []
        acc_base_list_all = []
        
        
        
        
        
        def RD_DPP_diveristy(eval_period):
            acc_base_list = np.zeros((Total_budget,2,EPOCH))
            dataset_base_X = dataset_gt_X_temp
            dataset_base_Y = dataset_gt_Y_temp
            cnn_eval_base = copy.deepcopy(cnn_gt_temp).to(DEVICE)
            pick_index2 = []
            Div_count = []
            for num_trans in range(0,Total_budget,eval_period):
                # print(num_trans)
                marginal_gain_list = np.zeros(len(train_loader))
                feats_all = np.zeros((len(train_loader),2880))+1e-8# len(feats)*num_class
                step_exlude =[]
                Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                R_0,R_c0 =code_diversity_empirical_old(Dataloader_base,cnn_eval_base,num_class=10,epsilon= 0.7)
                Div_count.append(R_0-R_c0)
                for step,(x,y) in enumerate(train_loader):
                    if step>=C_ini and (step not in pick_index2) and len(x)==BATCH_SIZE:
                        # print(step)
                        dataset_base_X_temp = torch.cat((dataset_base_X, x), 0)
                        dataset_base_Y_temp = torch.cat((dataset_base_Y, y), 0)
                        
                        # dataset_base_X_temp = torch.cat((dataset_gt_X_temp, x), 0)
                        # dataset_base_Y_temp = torch.cat((dataset_gt_Y_temp, y), 0)
                        
                        Dataloader_base_temp = Data.DataLoader(TensorDataset(dataset_base_X_temp,dataset_base_Y_temp.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                        
                        
                        
                        
                        R,R_c=code_diversity_empirical_old(Dataloader_base_temp,cnn_eval_base,num_class=10,epsilon= 0.7)
                        marginal_gain_list[step] =R-R_c #R-R_c #np.log(R-R_c)
                        Dataloader_base_temp2 = Data.DataLoader(TensorDataset(x,y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                        feats_all[step] = feat_empirical(Dataloader_base_temp2,cnn_eval_base)
                    
                    elif len(x)<BATCH_SIZE:
                        step_exlude.append(step)
                    else:
                        step_exlude.append(step)
                        
                        # div,_ = estimate_diversity_empirical(cnn_eval_base,Dataloader_base_temp)
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
                    Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                    cnn_eval_base = CNN(28,28,1,num_hidden=2).to(DEVICE)
                    optimizer_eval = torch.optim.Adam(cnn_eval_base.parameters(),lr=LR)
                    acc_train_,acc_test_ = train(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_eval_base,optimizer_eval,loss_func)
                    # acc_train_,acc_test_ =train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_eval_base,optimizer_eval,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                    acc_base_list[num_trans,0,:]=np.array(acc_train_)
                    acc_base_list[num_trans,1,:]=np.array(acc_test_)
                    print(np.mean(acc_test_[-20:]))
            Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
            cnn_base = CNN(28,28,1,num_hidden=2).to(DEVICE)
            optimizer_base = torch.optim.Adam(cnn_base.parameters(),lr=LR)
            acc_train_base,acc_test_base = train(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_base,optimizer_base,loss_func)
            print("the length of dpp is ")
            print(len(dataset_base_X))
            np.save(path+filename+'_div.npy',Div_count)
            return acc_base_list,acc_test_base
        
        
        
        def RD_DPP_two_stage(eval_period,bordline):
            acc_base_list = np.zeros((Total_budget,2,EPOCH))
            dataset_base_X = dataset_gt_X_temp
            dataset_base_Y = dataset_gt_Y_temp
            cnn_eval_base = copy.deepcopy(cnn_gt_temp).to(DEVICE)
            pick_index2 = []
            Div_count = []
            Flag = True
            
            Status_list = []
            
            for num_trans in range(0,Total_budget,eval_period):
                # print(num_trans)
                marginal_gain_list = np.zeros(len(train_loader))
                feats_all = np.zeros((len(train_loader),2880))+1e-8# len(feats)*num_class
                step_exlude =[]
                Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                R_0,R_c0 =code_diversity_empirical_old(Dataloader_base,cnn_eval_base,num_class=10,epsilon= 0.7)
                Div_count.append(R_0-R_c0)
                if (num_trans>0 and Div_count[-1]-Div_count[-2]>bordline and Flag == True and Div_count[-1] == max(Div_count)) or num_trans==0:
                    print('DPP select')
                    Status_list.append(1)
                    for step,(x,y) in enumerate(train_loader):
                        if step>=C_ini and (step not in pick_index2) and len(x)==BATCH_SIZE:
                            # print(step)
                            dataset_base_X_temp = torch.cat((dataset_base_X, x), 0)
                            dataset_base_Y_temp = torch.cat((dataset_base_Y, y), 0)
                            
                            # dataset_base_X_temp = torch.cat((dataset_gt_X_temp, x), 0)
                            # dataset_base_Y_temp = torch.cat((dataset_gt_Y_temp, y), 0)
                            
                            Dataloader_base_temp = Data.DataLoader(TensorDataset(dataset_base_X_temp,dataset_base_Y_temp.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                            
                            
                            
                            
                            R,R_c=code_diversity_empirical_old(Dataloader_base_temp,cnn_eval_base,num_class=10,epsilon= 0.7)
                            marginal_gain_list[step] =R-R_c #R-R_c #np.log(R-R_c)
                            Dataloader_base_temp2 = Data.DataLoader(TensorDataset(x,y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                            feats_all[step] = feat_empirical(Dataloader_base_temp2,cnn_eval_base)
                        
                        elif len(x)<BATCH_SIZE:
                            step_exlude.append(step)
                        else:
                            step_exlude.append(step)
                            
                            # div,_ = estimate_diversity_empirical(cnn_eval_base,Dataloader_base_temp)
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
                elif num_trans>0 and Div_count[-1]-Div_count[-2]<=bordline or Flag == False:
                    Flag = False
                    print('margin select')
                    Status_list.append(0)
                    loss_gain_list=  np.zeros(len(train_loader))
                    for step,(x,y) in enumerate(train_loader):
                        if step>=C_ini and (step not in pick_index2) and len(x)==BATCH_SIZE:    
                            y_predict = cnn_eval_base(x.to(DEVICE))[0]
                            prob_predict = F.softmax(y_predict,dim=1)
                            margin_pred = torch.sort(prob_predict,dim=1)[0]
                            margin_sum = torch.sum(margin_pred[:,-1]- margin_pred[:,-2]).item()
                            # loss =  1/loss_func(cnn_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                            # loss =  loss_func(cnn_eval_entropy(dataset_entropy_X_temp.to(DEVICE))[0],dataset_entropy_Y_temp.type(torch.LongTensor).to(DEVICE)).item()
                            loss_gain_list[step] = 1/margin_sum
                            
                            
                            # div,_ = estimate_diversity_empirical(cnn_eval_base,Dataloader_base_temp)
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
                    print('DPP')
                    Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
                    cnn_eval_base = CNN(28,28,1,num_hidden=2).to(DEVICE)
                    optimizer_eval = torch.optim.Adam(cnn_eval_base.parameters(),lr=LR)
                    acc_train_,acc_test_ = train(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_eval_base,optimizer_eval,loss_func)
                    # acc_train_,acc_test_ =train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_eval_base,optimizer_eval,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
                    acc_base_list[num_trans,0,:]=np.array(acc_train_)
                    acc_base_list[num_trans,1,:]=np.array(acc_test_)
                    print(np.mean(acc_test_[-20:]))
            Dataloader_base = Data.DataLoader(TensorDataset(dataset_base_X,dataset_base_Y.type(torch.LongTensor)),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
            cnn_base = CNN(28,28,1,num_hidden=2).to(DEVICE)
            optimizer_base = torch.optim.Adam(cnn_base.parameters(),lr=LR)
            acc_train_base,acc_test_base = train(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_base,optimizer_base,loss_func)
            print("the length of dpp is ")
            print(len(dataset_base_X))
            return acc_base_list,acc_test_base,Status_list
        # acc_train_base,acc_test_base = train_dis(Dataloader_base,val_loader,DEVICE,EPOCH,cnn_base,optimizer_base,loss_func,dataset_gt_X_temp,dataset_gt_Y_temp.type(torch.LongTensor),alpha = 0.1)
        
        # acc_train_gt,acc_test_gt = train(train_loader,val_loader,DEVICE,EPOCH,cnn_gt,optimizer_gt,loss_func)
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
            
            acc_entropy_list,acc_test_e = pick_by_entropy(eval_period)
            acc_margin_list,acc_test_margin = pick_by_min_margin(eval_period)
            
            
            acc_base_list,acc_test_base =pick_by_rate(eval_period)
            acc_base_list2,acc_test_base2 =pick_by_rate2(eval_period)
            acc_dpp_list,acc_test_dpp =[],[]
            acc_dpp_list_old,acc_test_dpp_old =RD_DPP_diveristy(eval_period)
            
            acc_a1_list,acc_test_a1,Status_list1 =RD_DPP_two_stage(eval_period,1)
            acc_a2_list,acc_test_a2,Status_list2 =RD_DPP_two_stage(eval_period,2)
            
            
            acc_entropy_list_all.append(acc_entropy_list)
            acc_margin_list_all.append(acc_margin_list)
            acc_base_list_all.append(acc_base_list)
            acc_dpp_list_all.append(acc_dpp_list)
            acc_base2_list_all.append(acc_base_list2)
            acc_dpp_old_list_all.append(acc_dpp_list_old)
            acc_adpative_1_list_all.append(acc_a1_list)
            acc_adpative_2_list_all.append(acc_a2_list)
            
            
            acc_entropy_list_f_all.append(acc_test_e)
            acc_margin_list_f_all.append(acc_test_margin)
            acc_base_list_f_all.append(acc_test_base)
            acc_dpp_list_f_all.append(acc_test_dpp)
            acc_base2_list_f_all.append(acc_test_base2)
            acc_dpp_list_f_old_all.append(acc_test_dpp_old)
            acc_adpative_1_f_all.append(acc_test_a1)
            acc_adpative_2_f_all.append(acc_test_a2)
            
    
        acc_rand_list,acc_test_rand_all = pick_by_random(5)
        acc_rand_list_all,acc_test_rand = pick_by_random_all(5)
        acc_rand_list_all,acc_test_rand = [],[]
        
        # plt.savefig('./results_rc/'+filename+str(np.random.randint(10000))+'all.png',bbox_inches='tight')
        # plt.savefig('./results_rc/'+filename+'all.png',bbox_inches='tight')
        acc_train_gt,acc_test_gt = train(train_loader,val_loader,DEVICE,EPOCH,cnn_gt,optimizer_gt,loss_func)
        results = [ acc_adpative_1_list_all,acc_adpative_2_list_all,acc_adpative_1_f_all,acc_adpative_2_f_all, \
                   acc_entropy_list_f_all,acc_entropy_list_all,acc_base_list_f_all,acc_base_list_all,acc_rand_list,acc_rand_list_all, \
                   acc_test_rand,acc_test_rand_all,acc_test_gt_temp,acc_dpp_list_f_all,acc_dpp_list_all,acc_base2_list_f_all,acc_base2_list_all,\
                       acc_dpp_list_f_old_all,acc_dpp_old_list_all,acc_margin_list_f_all,acc_margin_list_all,acc_test_gt,Status_list1,Status_list2]
        results = np.array(results,dtype=object)
        np.save(path+filename+'.npy', results)
    
    
    
    # np.mean(acc_test_base[-50:])
    # np.mean(acc_test_e[-50:])
    # np.mean(acc_test_rand[-50:])
