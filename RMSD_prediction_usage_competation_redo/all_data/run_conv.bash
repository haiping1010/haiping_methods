
input=$1
mkdir fold_${input%.txt}
cd fold_${input%.txt}
cat  ../$input  |  while read line
do

IFS=' ' read -r -a array <<< $line
##wget 'http://zinc15.docking.org/substances/'${array[0]}'.sdf'

base=${array[0]%.pdb}

cp -r  ../../../Uni-Dock/all_collect/group_*/$base'.mol2'  .


obabel  $base'.mol2' -O  $base'.pdb'  -d


echo $base

done

cd ..
