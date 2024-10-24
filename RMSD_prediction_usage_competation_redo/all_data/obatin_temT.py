
#complex1_L1_out  complex1_L2_out  complex1_L4_out  complex2_L3_out
import glob
import os


arr=glob.glob("complex?_L?_out/*_poc.pdb")


for line in arr:
     arr_tem=line.split('/')
     if  not os.path.isfile(arr_tem[0]+'.txt'):
          fw=open(arr_tem[0]+'.txt','w')
          fw.write(arr_tem[1]+'\n')
     else:
          fw=open(arr_tem[0]+'.txt','a')
          fw.write(arr_tem[1]+'\n')
           
fw.close()
