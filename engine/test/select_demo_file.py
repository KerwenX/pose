import os
import pickle
import json
import os
filename = '/home/aston/Desktop/Datasets/pose_data/scan2cad_image_alignments.json'

dataset_dir = '/home/aston/Desktop/Datasets/pose_data/ScanNOCS'

with open(filename,'r') as f:
    anno = json.load(f)

alignments = anno['alignments']

catid2_catname = {
    '03337140': 'cabinet',
    '02818832': 'bed',
    '04256520': 'sofa',
    '03001627': 'chair',
    '02747177': 'bin',
    '02933112': 'cabinet',
    # '03211117': 'display',
    '04379243': 'table',
    '02871439': 'bookcase',
    '02808440': 'bathtub'
}

catlist = list(catid2_catname.keys())
result = []

for scene in alignments:
    models = alignments[scene]
    label = True
    model_num=0
    label2 = True

    for model in models:
        model_num+=1
        if model['catid_cad'] not in catlist:
            label=False
            break
    meta_file = os.path.join(dataset_dir,scene+"_mata.txt")
    with open(meta_file,'rb') as f:
        temp = f.readlines()
        if len(temp)<3:
            label2 = False

    if label and model_num>=3 and label2:
        result.append(scene)

# for i in result:
#     print(i)
print(len(result))
result = [name+'\n' for name in result]
with open('demo_test.txt','w') as f:
    f.writelines(result)

print('hello world !')