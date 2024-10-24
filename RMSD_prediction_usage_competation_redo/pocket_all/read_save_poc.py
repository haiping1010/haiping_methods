import os
import sys
import numpy as np
import glob
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import pandas as pd
import re, os
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances

import random
from torch import nn








def  pdb_graph(pdbfile):
  cord = [None] * 3
  uniq=[]
  Pposition={}
  ResinameP={}
  Interface=[]
  residuePair=[]
  PL_pair=[]
  for line in open(pdbfile):
     tem_B=' '
     if len(line)>16:
        tem_B=line[16]
        line=line[:16]+' '+line[17:]
     #print(line)
     list_n = line.split()
     id = list_n[0]
     if id == 'ATOM' and tem_B !='B' and line.find(" HOH ") == -1:
        type = list_n[2]
        #print (line)
        if type == 'CA' and list_n[3]!= 'UNK':
            residue = list_n[3]
            atomname=list_n[2]
            type_of_chain = line[21:22]
            tem1=line[22:26].replace("A", "")
            tem2=tem1.replace("B", "")
            tem2=tem2.replace("C", "")

            #tem2=filter(str.isdigit, list_n[5])
            #atom_count = tem2+line[21:22]
            atom_count = line[4:11]+line[21:22]
            cord[0]=line[30:38]
            cord[1]=line[38:46]
            cord[2]=line[46:54]
            position = cord[0:3]
            Pposition[atom_count]=position
            ResinameP[atom_count]=line[17:26]
            #print atom_count,hash[residue[0:3]+atomname]

  for key1, value1 in Pposition.items():
     for key2, value2 in Pposition.items():
         if key2>key1:
            a = np.array(value1)
            a1 = a.astype(np.float64)
            b = np.array(value2)
            b1 = b.astype(np.float64)
            xx=np.subtract(a1,b1)
            tem=np.square(xx)
            tem1=np.sum(tem)
            out=np.sqrt(tem1)
            if out<5 :
                residuePair.append([ResinameP[key1],ResinameP[key2],out])
                uniq.append(ResinameP[key1])
                uniq.append(ResinameP[key2])
  return  residuePair,uniq,Pposition,ResinameP


import glob

arr_all=glob.glob("L*_poc.pdb")

dict_poc={}
for pdbfile in arr_all:
         print(pdbfile)
         residuePair,uniq,Pposition,ResinameP=pdb_graph(pdbfile)
         dict_poc[pdbfile]=[residuePair,uniq,Pposition,ResinameP]
import pickle


with open('dict_poc.pickle', 'wb') as f:
    pickle.dump(dict_poc, f)

with open('dict_poc.pickle', 'rb') as f:
    dict_poc_loaded = pickle.load(f)







