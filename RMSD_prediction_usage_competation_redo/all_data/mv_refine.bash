mkdir  refine_data
cat list_refine.txt | while read line

do

echo $line
mv  $line*   refine_data/



done
