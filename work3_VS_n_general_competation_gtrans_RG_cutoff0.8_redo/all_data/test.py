


frr=open('temT_1.txt','r')
arr_frr=frr.readlines()
pro_lig={}
arr_name=[]

for name in arr_frr:
     arr_name.append(name.strip())
     print name
     print name.strip()+'.dat'
     f = open(name.strip()+'.dat')
     t = f.read()
     print t


