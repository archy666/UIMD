import os
import torch
from utils import set_seed
from dataload import import_data
from scipy.io import loadmat
import numpy as np
from datetime import datetime
from model import IMCI
from config import get_args
from wapper import train_and_evaluate



set_seed(123)

""" Hyperparameter settings """
args = get_args()
data_path = args.data_path


""" Data sample count, dimension, and class count statistics """
temp = loadmat(data_path)
input_temp = torch.tensor(temp['X'][0][0]).float()
label_temp = torch.tensor(np.squeeze(temp['Y'])-1).long()
input_dims=[temp['X'][0][i].shape[1] for i in range(temp['X'][0].shape[0])]
class_num=len(np.unique(label_temp))
sample_number = temp['X'][0][0].shape[0]
view_number = len(input_dims)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


if __name__ == '__main__':
    for m_r_ in args.missing_rate:
        missing_rate = m_r_
        param_list = []
        for beta_i in args.beta:
            for lamb_i in args.lamb:
                for rho_i in args.rho:
                    param = [beta_i, lamb_i, rho_i]
                    results = []
                    net = IMCI(input_dims=input_dims, class_num=class_num).to(device)
                    train_dataset = import_data(data_path, missing_rate, random_state=1, train=True)
                    test_dataset = import_data(data_path, missing_rate, random_state=1,  train=False)
                    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=0)
                    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
                    report = train_and_evaluate(args, param, trainloader, testloader, net)
 
                    
        
        