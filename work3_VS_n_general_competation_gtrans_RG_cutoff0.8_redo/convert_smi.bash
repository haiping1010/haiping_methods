for name in *_ligand.mol2
do
	base=${name%.mol2}
	babel -imol2 $name  -osmi  $base'.smi'
	
done
