for name in *.sdf
do

base=${name%_ligand.sdf}
babel   $name  -O $base'_ligand_fixed.mol2'  -d

babel  $base'_ligand_fixed.mol2'  -O $base'_ligand_fixed.pdb'

done
