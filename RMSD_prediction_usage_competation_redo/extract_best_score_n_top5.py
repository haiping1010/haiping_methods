
import sys
import os

input_f=sys.argv[1]



# 读取数据
data = []
with open(input_f, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            score, name = line.split(',')
            score = float(score)
            data.append((score, name))

# 使用字典来存储分子及其对应的打分和构象ID列表
molecule_scores_ids = {}

# 分组数据
for score, name in data:
    molecule_name = name.split('_')[0]+'_'+name.split('_')[1]  # 假设名称的格式为"MoleculeID_conformationID"
    print (name)
    conformation_id = name.split('_')[2].split('.')[0]

    #print (conformation_id)
    
    if molecule_name not in  molecule_scores_ids.keys():
        molecule_scores_ids[molecule_name] = []
    molecule_scores_ids[molecule_name].append((score, conformation_id))

# 选择每个分子的最好的三个打分及其对应的构象ID
top_scores_ids = {}
for molecule_name, score_id_pairs in molecule_scores_ids.items():
    print (score_id_pairs)
    sorted_score_id_pairs = sorted(score_id_pairs, key=lambda x: x[0])  # 按打分排序
    top_scores_ids[molecule_name] = sorted_score_id_pairs[:5]  # 取最小的三个值及其对应的构象ID
    print (sorted_score_id_pairs[:5])




# 输出结果

output_file = input_f.rsplit(".txt", 1)[0] + "_top_scores5.txt"
i=0
with open(output_file, 'w') as fw:
    for molecule_name, scores_ids in top_scores_ids.items():
        #fw.write(f"{molecule_name}\t")
        #fw.write("Top 3 scores and their conformations:\n")
        for score, id in scores_ids:
            index=i%5
            index_n='model_'+str(index+1)
            fw.write(f"{score}\t{molecule_name}_{id}\t{index_n}")
            fw.write("\n")
            i=i+1


