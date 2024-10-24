for name in *_ligand.sdf
do

	base=${name%_ligand.sdf}
	python  convert_sdf_smi.py  $base
        grep -v 'SMILE' $base'_ligand.smi' > $base'_ligand_n.smi'
	mv $base'_ligand_n.smi'  $base'_ligand.smi'


done
