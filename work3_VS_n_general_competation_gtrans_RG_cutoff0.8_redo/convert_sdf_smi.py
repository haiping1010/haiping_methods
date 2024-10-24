from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesWriter
import sys
input1=sys.argv[1]
mols = [mol for mol in Chem.SDMolSupplier(input1+'_ligand.sdf') if mol != None]
# make writer object with a file name.
writer = SmilesWriter(input1+'_ligand.smi')
#Check prop names.
 
for mol in mols:
     writer.write( mol )
     
writer.close()
