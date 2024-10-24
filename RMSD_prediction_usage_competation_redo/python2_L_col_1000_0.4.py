import sys
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import KMeans
import glob
def one_hot(i):
    m = np.zeros(41, 'uint8')
    m[i] = 1
    return m

def one_hot_atom(i):
    m = np.zeros(84, 'uint8')
    m[i] = 1
    return m

def MapResToOnehot(atom1,atom2):
   code={}
   codelig={}
   seq=('C','CA','CB','CC','CK','CM','CN','CQ','CR','CT','CV','CW','C*','F','H','HC','H1','H2','H3','HA','H4','H5','HO','HS','HW','HP','N','NA','NB','NC','N2','N3','N*','O','OW','OH','OS','O2','P','S','SH')
##   seqatom=("c","c1","c2","c3","ca"," n","n1","n2","n3","n4","na","nh","no","f","cl","br","i","h1","h2","h3","h4","h5","n","nb","nc","nd","sx","sy","o","oh","os","s2","sh","ss","s4","s6","hc","ha","hn","ho","hs","hp","p2","p3","p4","p5","cc","cd","ce","cf","cp","cq","cu","cv","cx","cy","pb","pc","pd","pe","pf","px","py")
   seqatom=("c","cs","c1","c2","c3","ca","cp","cq","cc","cd","ce","cf","cg","ch","cx","cy","cu","cv","cz","h1","h2","h3","h4","h5","ha","hc","hn","ho","hp","hs","hw","hx","f","cl","br","i","n","n1","n2","n3","n4","na","nb","nc","nd","ne","nf","nh","no","ns","nt","nx","ny","nz","n+","nu","nv","n7","n8","n9","o","oh","os","ow","p2","p3","p4","p5","pb","pc","pd","pe","pf","px","py","s","s2","s4","s6","sh","ss","sx","sy","B")
   

   index=0
   for name in seq:
              #print (name)
              code[name]=one_hot(index)
              index=index+1
   index=0
   for name in seqatom:
              #print (name)
              codelig[name]=one_hot_atom(index)
              index=index+1         
   ######
   ##codelig['DU']="00000000000000000000000000000000000000000000000000000000000000000000000000000000000"       
   return (np.concatenate((code[atom1],codelig[atom2])))
#print (MapResToOnehot("ARG","VAL"))

#-------------------------------------------------
#################read  topology
f=open('topol.top', 'r')
hash={}
lines=f.readlines()
index=0
for line in lines:
    if line.startswith('[ atoms ]') :
         #print (line.strip()) 
          index=1

    if line.startswith('[ bonds ]') :
          index=0
    if index==1 and line[0] != '[' and  line.find('[')<0 and line[0] != ';'  and line != '\n':
          arr=line.split()
          atom_name=arr[4]
          ############atom name in the pdb and toplogy has slightly difference       
          if len(arr[4])==4 and atom_name[0]=='H':
              arr[4] = atom_name[-1] + atom_name[0:3]
          #print arr[1]
          hash[arr[3]+arr[4]]=arr[1]
 
          #print  line
          #print index
#################read topology end

cord = [None] * 3
visited = {}
Pposition={}
Lposition={}
ResinameP={}
ResinameL={}
Interface=[]
residuePair=[]
if len(sys.argv) <1 :
   print("python python2_L.py xxx")
filebase=sys.argv[1]
##filebase=file.replace(".pdb","")

#filebase="1flr"
#print(file)
print(filebase)

for line in open(filebase[0:4]+'_processed.pdb'):
##for line in open(filebase+'_pocket.pdb'):
    tem_B=' '
    if len(line)>16:
       tem_B=line[16]
       line=line[:16]+' '+line[17:]
    #print(line)
    list = line.split()
    id = list[0]
    if id == 'ATOM' and tem_B !='B' and line.find(" HOH ") == -1:
        type = list[2]
        #print (line)
        #if type == 'CA' and list[3]!= 'UNK':
        if  list[3]!= 'UNK':
            residue = list[3]
            atomname=list[2]
            type_of_chain = line[21:22]
            tem1=line[22:26].replace("A", "")
            tem2=tem1.replace("B", "")
            tem2=tem2.replace("C", "")

            #tem2=filter(str.isdigit, list[5])
            #atom_count = tem2+line[21:22]
            atom_count = line[4:11]+line[21:22]
            cord[0]=line[30:38]
            cord[1]=line[38:46]
            cord[2]=line[46:54]
            position = cord[0:3]
            Pposition[atom_count]=position
            ResinameP[atom_count]=hash[residue[0:3]+atomname]
            ResinameP[atom_count]=residue
            #print atom_count,hash[residue[0:3]+atomname]
index_nn=0            
ligarr=glob.glob(filebase[0:4]+"*_ligand_n*.mol2")
print (filebase[0:9])
for line in open(ligarr[0]):
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
            #print (key1)
            #print ResinameP[key1]
            a = np.array(value1)
            a1 = a.astype(np.float)
            b = np.array(value2)
            b1 = b.astype(np.float)
            xx=np.subtract(a1,b1) 
            tem=np.square(xx)
            tem1=np.sum(tem)
            out=np.sqrt(tem1)
            #print (tem1)
            if out<4 :
                #print (ResinameHL[key1],ResinameHL[key2])
                #print (a,b,out)
                #print (ResinameP[key1])              
                residuePair.append([ResinameP[key1],ResinameL[key2]])
                Interface.append(a1)
                #print (a1)  
#---------------------------------------------------------------------                
#print (antiface)              
kmeans = KMeans(n_clusters=5, random_state=0).fit_predict(Interface)
#print (kmeans)
indexx=0
kmeans1=kmeans.tolist()

index_n=0
for i in range(5):
   #print(kmeans)
   g=np.argwhere(kmeans==i)
   ##g=kmeans.where(q==i) 
   ##print (g)
   g1=g.reshape(-1)

   for groupid in g1:
      index_n=index_n+1
      #if index_n<=1000:
           #print(residuePair[groupid][0],residuePair[groupid][1])
		   
		   
#------------------------------------------------------
#print (antiface)              
kmeans = KMeans(n_clusters=5, random_state=0).fit_predict(Interface)
#print (kmeans)
indexx=0
kmeans1=kmeans.tolist()
foo = open(filebase + "_learn_aa1000_0.4.txt", "w")

index_n=0
for i in range(5):
   #print(kmeans)
   g=np.argwhere(kmeans==i)
   ##g=kmeans.where(q==i) 
   ##print (g)
   g1=g.reshape(-1)

   for groupid in g1:
      index_n=index_n+1
      if index_n<=1000:
           writout=MapResToOnehot(residuePair[groupid][0],residuePair[groupid][1])
           # foo = open("3HFM_learn.txt", "a")
           arr_tem=np.array_str(writout)
           for code in  writout:
              #print (str(code))
              foo.write(str(code))
              # print(code)
              #foo.write(code)
           foo.write('\n')
while index_n<1000:
         index_n=index_n+1
         foo.write("00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000\n")


