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
filebase=sys.argv[1]
print(filebase)
for line in open(filebase+'_w.pdb'):
    tem_B=' '
    if len(line)>16:
       tem_B=line[16]
       line=line[:16]+' '+line[17:]
    #print(line)
    list = line.split()
    id = list[0]
    if id == 'ATOM' and tem_B !='B':
        type = list[2]
        #if type == 'CA' and list[3]!= 'UNK':
        if  list[3]!= 'UNK':
            residue = list[3]
            type_of_chain = line[21:22]
            tem1=line[22:26].replace("A", "")
            tem2=tem1.replace("B", "")
            tem2=tem2.replace("C", "")

            #tem2=filter(str.isdigit, list[5])
            atom_count = tem2+line[21:22]
            list[6]=line[30:38]
            list[7]=line[38:46]
            list[8]=line[46:54]
            position = list[6:9]
            Pposition[atom_count]=position
            list[4]=line[21:22]
            list[5]=line[22:26].replace(' ','')
            ResinameP[atom_count]=residue+list[5]+list[4]
            resindex=residue+list[5]
            Atomline.append(line)
index_nn=0            
for line in open(filebase+'_ligand.mol2'):
    tem_B=' '
    line=line.strip()
    #print(line)
    if line == "@<TRIPOS>ATOM":
        index_nn=1
        #print(line)
    if line == "@<TRIPOS>BOND":
        index_nn=0 
    if index_nn==1 and line != "@<TRIPOS>ATOM":
            list = line.split()
            #tem2=filter(str.isdigit, list[5])
            atom_count = list[0]+list[5]
            position = list[2:5]
            Lposition[atom_count]=position
            ResinameL[atom_count]=list[5]
			
			
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
            #print (tem1)
            if out<10 :
                residuePair.append(ResinameP[key1])
                #print (ResinameP[atom_count])  
#---------------------------------------------------------------------                
#print (antiface)              
foo = open(filebase + "_poc.pdb", "w")

for  value1 in Atomline:
    for value2 in residuePair:    
        list2 = value1.split()
        list2[4]=value1[21:22]
        list2[5]=value1[22:26].replace(' ','')
        resnameId=list2[3]+list2[5]+list2[4]
        #print value2, resnameId
        if  value2 == resnameId:
              print (key1, value2)
              foo.write(value1 )
              break

