
cd  ../

for name in L*_prepared
do

cd $name

for folder in L*
do

cd $folder

for lig in ligand_*.pdb
do
cp -r  $lig     ../../collect/$folder'_'$lig
done

cp -r  protein_aligned.pdb   ../../collect/$folder'_protein_aligned.pdb'


cd  ..


done



cd ..
done
