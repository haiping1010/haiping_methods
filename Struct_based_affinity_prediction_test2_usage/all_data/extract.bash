
for name in    L????_ligand_*.mol2   
do
base=${name:0:5}

echo $base
#babel -isdf $base'_ligand.sdf'  -omol2  $base'_ligand_n.mol2'

grep "^ATOM\|^TER\|^END" $base'_protein_aligned.pdb'  > $base'_w.pdb'

nohup python extract_pocket.py  $name  $base &

sleep  2s


done
