import sys
import numpy as np

Pposition={}
Lposition={}
ResinameP={}
ResinameL={}
Atomline=[]
residuePair=[]



if len(sys.argv) <1 :
   print("python python2_L.py xxx")
ligfile=sys.argv[1]
filebase=sys.argv[2]
print(filebase)
for line in open(filebase+'_w.pdb'):
    tem_B=' '
    if len(line)>16:
       tem_B=line[16]
       line=line[:16]+' '+line[17:]
    #print(line)
    list_n = line.split()
    id = list_n[0]
    if id == 'ATOM' and tem_B !='B':
        type = list_n[2]
        #if type == 'CA' and list_n[3]!= 'UNK':
        if  list_n[3]!= 'UNK':
            residue = list_n[3]
            type_of_chain = line[21:22]
            tem1=line[22:26].replace("A", "")
            tem2=tem1.replace("B", "")
            tem2=tem2.replace("C", "")

            #tem2=filter(str.isdigit, list_n[5])
            #print line[4:11]
            atom_count = line[4:11]+tem2+line[21:22]
            list_n[6]=line[30:38]
            list_n[7]=line[38:46]
            list_n[8]=line[46:54]
            position = list_n[6:9]
            Pposition[atom_count]=position
            list_n[4]=line[21:22]
            list_n[5]=line[22:26].replace(' ','')
            ResinameP[atom_count]=residue+list_n[5]+list_n[4]
            resindex=residue+list_n[5]
            Atomline.append(line)
index_nn=0            
for line in open(ligfile):
    tem_B=' '
    line=line.strip()
    #print(line)
    if line == "@<TRIPOS>ATOM":
        index_nn=1
        #print(line)
    if line == "@<TRIPOS>BOND":
        index_nn=0 
    if index_nn==1 and line != "@<TRIPOS>ATOM":
            list_n = line.split()
            #tem2=filter(str.isdigit, list_n[5])
            atom_count = list_n[0]+list_n[5]
            position = list_n[2:5]
            Lposition[atom_count]=position
            ResinameL[atom_count]=list_n[5]
			
			
#-------------------------------------------------

for key1, value1 in Pposition.items():
   #print ( key1)
   for key2, value2 in Lposition.items():
            #print (ResinameE[key], 'corresponds to', value)
            ##distance=pow(value1[0]-value2[0])
            #print (value2)
            a = np.array(value1)
            a1 = a.astype(np.float)
            b = np.array(value2)
            b1 = b.astype(np.float)
            xx=np.subtract(a1,b1) 
            tem=np.square(xx)
            tem1=np.sum(tem)
            out=np.sqrt(tem1)
            #print (out)
            if out<8 :
                residuePair.append(ResinameP[key1])
                #print (ResinameP[atom_count])  
#---------------------------------------------------------------------                
#print (antiface)              
foo = open(ligfile.replace('.mol2','') + "_poc.pdb", "w")

for  value1 in Atomline:
    for value2 in residuePair:    
        list_n2 = value1.split()
        list_n2[4]=value1[21:22]
        list_n2[5]=value1[22:26].replace(' ','')
        resnameId=list_n2[3]+list_n2[5]+list_n2[4]
        #print value2, resnameId
        if  value2 == resnameId:
              #print (key1, value2)
              foo.write(value1 )
              break

