import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, global_max_pool as gmp  
from torch_geometric.data import InMemoryDataset, DataLoader

TRAIN_BATCH_SIZE = 100

# GCN based model
class GCNNet(torch.nn.Module):
    #def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=21, output_dim=128, dropout=0.2):
    def __init__(self, n_output=1, n_filters=64, embed_dim=1280,num_features_xd=75,edge_dim_xd=20, output_dim=1280, dropout=0.2):
        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        print (str(num_features_xd)+'xxxxxx')
        self.conv1 = TransformerConv(num_features_xd, num_features_xd ,heads=5, concat=True, dropout=0.1, edge_dim=edge_dim_xd)
        self.conv2 = TransformerConv(num_features_xd*5, num_features_xd*10,heads=10, concat=True, dropout=0.1,edge_dim=edge_dim_xd)
        #self.conv3 = TransformerConv(num_features_xd*2*4, num_features_xd * 4,heads=4,concat=True, dropout=0.1,edge_dim=edge_dim_xd)
        #self.conv4 = TransformerConv(num_features_xd*4*4, num_features_xd ,heads=4,concat=True, dropout=0.1,edge_dim=edge_dim_xd)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*10, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(dropout)


        # combined layers
        self.fc1 = nn.Linear(output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data, TRAIN_BATCH_SIZE, device):
        # get graph input
        x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch,data.edge_attr 
        #print (edge_attr)
        #print (data.batch)
        #print (x)
        #print ('yyyyyyyyyyyyyyyyyyyyyyyyyyy')
        # get protein input
        #target = data.target
        #print (data.name)
        #print (target)

        x = self.conv1(x, edge_index,edge_attr)
        x = self.relu(x)

        x = self.conv2(x, edge_index,edge_attr)
        x = self.relu(x)

        #x = self.conv3(x, edge_index,edge_attr)
        #x = self.relu(x)

        #x = self.conv4(x, edge_index,edge_attr)
        #x = self.relu(x)



        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        #x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        ##embedded_xt = self.embedding_xt(target)
        ##conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        ##xt = conv_xt.view(-1, 32 * 121)
        ##xt = self.fc1_xt(xt)
        #pocket graphic

        #  combination
        # concat

        # add some dense layers
        #xc = self.fc1(xc)
        xc=self.fc1(x)
        xc = self.relu(xc)
        #xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out



