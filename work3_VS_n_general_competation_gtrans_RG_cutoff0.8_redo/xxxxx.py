import os
import sys
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import KMeans
import glob
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def residue_features(resname):
    seq=("ALA","ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",'Unkown')
    return np.array(one_of_k_encoding_unk(resname,seq))



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

compound_iso_smiles = []
import glob
arr_name=glob.glob("????_ligand.smi")

smile_graph = {}

for name in  arr_name:
   fr=open(name,'r')
   arr=fr.readlines()
   linearr=arr[0].split('\t')
   compound_iso_smiles.append(linearr[0])
   #print (linearr[0])
   #print (linearr[1])
   smile_graph[name[0:4]] =smile_to_graph(linearr[0])
    


cord = [None] * 3

Pposition={}
ResinameP={}
Interface=[]
residuePair=[]


uniq=[]

def  pdb_graph(pdbfile):
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
            a1 = a.astype(np.float)
            b = np.array(value2)
            b1 = b.astype(np.float)
            xx=np.subtract(a1,b1)
            tem=np.square(xx)
            tem1=np.sum(tem)
            #out=np.sqrt(tem1)
            #if out<5 :
            if tem1<np.square(5):
                residuePair.append([ResinameP[key1],ResinameP[key2]])
                uniq.append(ResinameP[key1])
                uniq.append(ResinameP[key2])
                #Interface.append(a1)
  uniq_n=list(set(uniq))
  my_dict = {}
  for index, item in enumerate(uniq_n):
        my_dict[item] = index

  edges_p=[]
  features=[]
  for i in residuePair:
     #print ( my_dict[i[0]], my_dict[i[1]])
     edges_p.append([my_dict[i[0]], my_dict[i[1]]])
  for index, item in enumerate(uniq_n):
        #print (item)
        feature = residue_features(item[0:3])
        features.append( feature / sum(feature) )
  c_size=len(uniq_n)
  #print (c_size)
  return  c_size,features,edges_p

pocket_graph = {}
import glob

##arr_name=glob.glob("????_pocket.pdb")
frr=open('tem.txt','r')
arr_frr=frr.readlines()
pro_lig={}
arr_name=[]
for name in arr_frr:
     print (name)
     arr=name.split()
     pro_lig[arr[0]] = arr[1]
     arr_name.append(arr[0]+'_poc.pdb')

