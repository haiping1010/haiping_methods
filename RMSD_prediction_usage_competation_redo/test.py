from rdkit import Chem
import glob

def  read_sdf(molpath):
        print (molpath)
        mol = Chem.MolFromMolFile(molpath,removeHs=True)
        print (mol,mol.GetNumAtoms())

all_arr=glob.glob("all_data/*_fixed.sdf")
for name in all_arr:
    try :
       read_sdf(name)
    except:
        print('wrong')


