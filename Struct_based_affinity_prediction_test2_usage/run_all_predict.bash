for name in  data1/processed/L_P_train_*.pt
do

base=${name:16 }
base_n=${base%.pt}
echo $base
nohup python  training_nn3_load_name.py  $base_n > $base_n'_predict.log'  2>&1&

sleep 60s

done



