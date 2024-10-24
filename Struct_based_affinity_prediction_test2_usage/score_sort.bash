
cat  output_??_n.txt  > all_out.list

sort -g -rk 1,1 all_out.list  > all_out.sort

awk -F ',' ' $1 >= 7.8 ' all_out.sort  >  all_out_select.sort

