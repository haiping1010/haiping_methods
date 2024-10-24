for name in  6*_protein.pdb
do

base=${name%_protein.pdb}

grep '^ATOM\|^TER\|^END'  $name  > $base'_w.pdb'

nohup python  extract_pocket.py  $base  &
sleep  1s


done
