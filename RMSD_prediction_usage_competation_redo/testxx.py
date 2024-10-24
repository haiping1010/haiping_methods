import os
import sys
import numpy as np
import glob
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

from rdkit import Chem
mol = Chem.MolFromPDBFile('all_data/fold_temT_1/10gs_ligand_out11.pdb', removeHs=not False)
idx=1
name='10gs_ligand_out11.pdb'
mol = Chem.MolFromPDBFile('all_data/'+'fold_temT_'+str(idx)+'/'+name, removeHs=not False)
