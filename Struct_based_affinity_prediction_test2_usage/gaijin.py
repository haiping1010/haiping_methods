import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, pocket_graph=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, pocket_graph, y, smile_graph)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, pocket_graph, y, smile_graph):
        data_list = []
        data_len = len(xd)
        
        def process_sample(i):
            filename = xd[i]
            labels = y[i]
            fr = open('all_data/' + str(filename[0:4]) + '_ligand.smi', 'r')
            arr = fr.readlines()
            linearr = arr[0].split('\t')
            c_size, features, edge_index = smile_to_graph(linearr[0])
            
            GCNData = Data(x=torch.Tensor(features),
                           edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                           y=torch.FloatTensor([labels]))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            
            c_size1, features1, edge_index1 = pdb_graph('all_data/' + filename[0:4] + '_poc.pdb')
            GCNData.name = filename[0:4]
            GCNData.target = Data(x=torch.Tensor(features1),
                                  edge_index=torch.LongTensor(edge_index1).transpose(1, 0))
            return GCNData
        
        with ThreadPoolExecutor() as executor:
            data_list = list(executor.map(process_sample, range(data_len)))
        
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# 示例代码，假设其他部分代码如 smile_to_graph 和 pdb_graph 已经定义
train_Y = np.array(train_Y)
train_data = TestbedDataset(root='data1', dataset='L_P_train_' + str(idx), xd=arr_name, pocket_graph=pocket_graph, y=train_Y, smile_graph=smile_graph):wq

