for folder in  complex*_L*_out
do


cd  $folder

for name in *.pdbqt
do

base=${name%.pdbqt}
#obabel  $name   -O  $base'.mol2'  -m  -d



for filename in $base*'.mol2'
do

base_n=${filename%.mol2}
obabel  $filename   -O  $base_n'.pdb' 

done



done


cd  ..



done

