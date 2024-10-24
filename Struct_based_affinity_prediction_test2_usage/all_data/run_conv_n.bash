
for name in L*ligand*.pdb
do
base=${name%.pdb}

nohup obabel  $base'.pdb' -O  $base'.mol2'  -d &
sleep 0.3s


echo $base

done

