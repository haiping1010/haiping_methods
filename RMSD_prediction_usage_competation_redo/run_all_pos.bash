#conda activate py3
#for name in   all_data/temT_*.txt
#for xx in 132  141  144  157 166 25 36  45   49    57    58  151   156  1 40   47   76    80  
#for  xx  in  64  65 6 70 71 72 73  74  75  77  78  79  7  81  87  88  89 8 90 91 92 93 94
#for xx in 138 161 16 26 34 35 38 43 44 46 48 4 50 51 52 53 54 55 56 61 62 63
#########for xx in  {1..101}

for xx in  complex1_L4_out.txt  complex2_L3_out.txt
do


nohup python  read_smi_protein_nnn_upgrade.py  $xx  > 'outT_'$xx'.log' 2>&1&
##sleep 120s

#python  read_smi_protein_nnn_upgrade.py  $id  > 'outT_'$id'.log'
 
#sleep 0.1s
done
