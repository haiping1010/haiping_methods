
for name in ????_protein.pdb
do

base=${name%_protein.pdb}
grep "^ATOM\|^TER\|^END"  $name  > $base'_w.pdb'


done
