
for name in   complex?_L{3..4}_out  

do

base=${name%_out}

cd  $name


for filex in *.mol2
do

nohup python  ../extract_pocket_n.py    ../$base'_poc_w.pdb'  $filex  &
sleep  0.1s



done

cd  ..



done
