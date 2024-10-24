for name in 9*.sdf
do

base=${name%_ligand.sdf}
babel   $name  -O $base'_ligand_n.mol2'  -d

babel  $base'_ligand_n.mol2'  -O $base'_ligand_n.pdb'

done
