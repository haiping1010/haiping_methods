import os
import sys
import numpy as np
import glob




fr=open('collect_final_score_good.txt','r')
arr_n=fr.readlines()

dict_label={}
for line in  arr_n:
   arr_tem=line.split('\t')
   dict_label[arr_tem[0].replace('.mol2','.pdb')] = arr_tem[1]

import json

# Save dict_label to a JSON file
with open('dict_label.json', 'w') as f:
    json.dump(dict_label, f)

print("dict_label saved successfully.")


import json

# Load dict_label from JSON file
with open('dict_label.json', 'r') as f:
    dict_label_loaded = json.load(f)









