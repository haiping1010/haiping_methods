for name in ????_ligand.smi
do

base=${name%_ligand.smi}
cp  -r  /data/bcup_all/torch/Graph_PDbind_net/pdbbind2019_all/$base/$base'_ligand.sdf'  .

done
