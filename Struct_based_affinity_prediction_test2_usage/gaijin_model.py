import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data

# Define a simple graph data
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 6, 7],
                           [1, 0, 2, 1, 3, 2, 5, 4, 7, 6]])
x = torch.randn(8, 16)  # 8 nodes, each with 16 features
data = Data(x=x, edge_index=edge_index)

# Define a model using TransformerConv
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = TransformerConv(in_channels=16, out_channels=32, heads=4, concat=True, dropout=0.1)
        self.conv2 = TransformerConv(in_channels=32, out_channels=64, heads=8, concat=True, dropout=0.1)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        return x

# Initialize the model
model = MyModel()

# Perform a forward pass
output = model(data)
print(output.shape)  # Example output shape
