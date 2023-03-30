# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:42:26 2022

@author: Xiwen Chen
"""

# date:2020/9/10
import numpy as np
import math,time
import matplotlib.pyplot as plt
'''
algorithm is using: 
Fast greedy map inference for determinantal point process to improve recommendation diversity
'''
def dpp(kernel_matrix, max_length, epsilon=1E-20): 
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
        # print(di2s)
        selected_item = np.argmax(di2s)
        # print(di2s[selected_item])
        if di2s[selected_item] < epsilon:
            
            break
        selected_items.append(selected_item)
    return selected_items




if __name__ == '__main__':
    
    
    dis_list = ['Uniform', 'Beta', 'Binomial','Exponential','Rayleigh','Poisson']
    
    # dis = 'Poisson'
    
    for dis in dis_list:
        print(dis)
        
        
        
        
        item_size = 200
        feature_dimension = 500
        alpha = feature_dimension/0.5  # you can change to any number
        max_length = min(item_size,feature_dimension)
         
       
        if dis== 'Gaussian': 
            feature_vectors = np.random.randn(item_size, feature_dimension)
        elif dis== 'Uniform': 
            feature_vectors = np.random.rand(item_size, feature_dimension)
        elif dis == 'Beta': #a=1 b=5
            feature_vectors = np.random.beta(1, 5,size=(item_size, feature_dimension))
        elif dis == 'Binomial': #n trails # p
            feature_vectors = np.random.binomial(10, .5,size=(item_size, feature_dimension)).astype(float)
        elif dis == 'Exponential': # 1
            feature_vectors = np.random.exponential(size=(item_size, feature_dimension))
        elif dis == 'Rayleigh': # 1
            feature_vectors = np.random.rayleigh(size=(item_size, feature_dimension))
        elif dis == 'Poisson':
            feature_vectors = np.random.poisson(size=(item_size, feature_dimension)).astype(float)
        else:
            # define your distribution
            pass
            
            
            
        similarities = np.dot(feature_vectors, feature_vectors.T)
        kernel_matrix = similarities
        
        # kernel_matrix[2:4,2:4]
         
        print('kernel matrix generated!')
        t = time.time()
        result = dpp(kernel_matrix, max_length)
        print(len(result))
        # print('algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
        
        
    
    # np.linalg.det(kernel_matrix)
    
    
    
    
    # np.random.randn(288*128,288*128).T*(np.random.randn(4,288*128))
    
    
        value = []
        for i in range(max_length):
            
            index = result[0:i+1]
            
            selected_feat = np.take(feature_vectors,index,axis=(0))
            
            var_feat = np.var(selected_feat,axis = 0)
            
            upper_bound = np.sum(np.log(alpha*var_feat+1))
            value.append(upper_bound)
            
        # plt.figure()
        
        fig=plt.figure(dpi=500)
        font1={'font.family':'serif',
                'font.serif':'Times New Roman',
                'font.style':'normal',
                'font.weight':'bold'}#or large,small
        
               
        plt.rcParams.update(font1)
        plt.rcParams['text.usetex'] = False
        
        ax = fig.add_subplot(111)
        plt.plot(range(1,max_length+1),value)
        plt.ylabel('Upper. of Div.')
        plt.xlabel('Number of samples')
        plt.title(f'{dis}')
        
        plt.rc('xtick', labelsize=18) 
        plt.rc('ytick', labelsize=18) 
        plt.rc('axes', labelsize=18) 
        plt.rc('axes', titlesize=18) 
        plt.rc('legend', fontsize=12)
        
        
        # plt.savefig(f'out/{dis}_{feature_dimension}', bbox_inches = 'tight')



