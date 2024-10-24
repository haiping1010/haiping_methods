import shutil, random, os
dirpath='/home/zhanghaiping/program/torch/Graph_PDbind_net/refined-set_test/all_pdb_all_n/all_data'
#filenames = random.sample(os.listdir(dirpath), 4000)
import glob
os.chdir(dirpath)
aa=glob.glob("????_protein.pdb")
##filenames = random.sample(os.listdir(dirpath), 3)
#filenames = random.sample(aa, 3)
fw=open("tem.txt", "w")
print (fw.name)

for pname in aa:
  filenames = random.sample(aa, 3)
  for lname in filenames:
      while pname == lname:
          tem=random.sample(aa, 1)
          lname=tem[0]      
      fw.write(pname.replace('_protein.pdb','') +"  "+ lname.replace('_protein.pdb','')+"\n")
#      fw.write(str(pname))
#      fw.write("1")
   #srcpath = os.path.join(dirpath, fname)
   #disfile=os.path.join(destDirectory, fname)
   #shutil.copyfile(srcpath, disfile)

fw.close()
