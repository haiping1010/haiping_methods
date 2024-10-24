from rdkit import Chem
import glob

def  read_sdf(molpath):
        print (molpath)
        #mol = Chem.MolFromMol2File(molpath, removeHs=not False)
        mol =  Chem.MolFromPDBFile(molpath, removeHs=not False)
        print (mol,mol.GetNumAtoms())

all_arr=glob.glob("all_data/*_fixed.pdb")
for name in all_arr:
    try :
       read_sdf(name)
    except:
        print('wrong')


