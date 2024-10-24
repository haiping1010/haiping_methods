import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
#from  read_smi_protein import *
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
from utils import *

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


#modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
#model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 2
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


'''
# Main program: iterate over different datasets
for dataset in 'L_P':
    #print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data1/processed/' + 'L_P_train.pt'

    if ((not os.path.isfile(processed_data_file_train))) :
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset2(root='data1', dataset='L_P_train')
        #train_data, slices = torch.load(processed_data_file_train)
        #print (train_data)
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        print (train_loader)
        print('Training on {} samples...'.format(len(train_loader.dataset)))
'''

import glob 
from torch.utils.data.dataset import Dataset, ConcatDataset

dataset=TestbedDataset2(root='data1', dataset='L_P_train_neg_1')

for name in range(2,20):
    print (name)
    train_data = TestbedDataset2(root='data1', dataset='L_P_train_neg_'+str(name))
    dataset=dataset + train_data

train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

for data in train_loader:
            #x, edge_index, batch = data.x, data.edge_index, data.batch
            #print(data.batch)
            print (len(data))



