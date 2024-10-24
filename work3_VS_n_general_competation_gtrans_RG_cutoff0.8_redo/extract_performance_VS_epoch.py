

fr=open('output_train.log','r')

arr=fr.readlines()


print 'epoch:'+'\t'+'rmse:'+'\t'+'mse:'+'\t'+'pearson'+'\t'+'spearman'+'\t'+'ci'+'\n'
for line in arr:
    if  line.startswith('epoch:'):
        tem_arr=line.split(':')
        out_line=tem_arr[1].strip()
    if  line.startswith('rmse:'):
        tem_arr=line.split(':')
        out_line=out_line+'\t'+tem_arr[1].strip()
    if  line.startswith('mse:'):
        tem_arr=line.split(':')
        out_line=out_line+'\t'+tem_arr[1].strip()
    if  line.startswith('pearson'):
        tem_arr=line.split()
        out_line=out_line+'\t'+tem_arr[1].strip()
    if  line.startswith('spearman'):
        tem_arr=line.split()
        out_line=out_line+'\t'+tem_arr[1].strip()
    if  line.startswith('ci'):
        tem_arr=line.split()
        out_line=out_line+'\t'+tem_arr[1].strip()
        print out_line
