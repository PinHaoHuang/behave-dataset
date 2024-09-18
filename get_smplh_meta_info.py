from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer 
import json
import torch
import pickle as pkl
import numpy as np

model_root = "support_data/mano"

  
smpl = SMPL_Layer(center_idx=0, gender="male", num_betas=10,
                    model_root=str(model_root), hands=True)


with open('support_data/smplh_kintree_parents.json', 'w') as f:
    json.dump(smpl.kintree_parents, f)


seq_root = "/home/phuang/sv871514lx_data/BEHAVE/sequences" # replace this with the root path of behave sequences
frame = "Date01_Sub01_chairwood_hand/t0003.000"
param_file = f"{seq_root}/{frame}/person/fit02/person_fit.pkl"


smpl_dict = pkl.load(open(param_file, 'rb'))
p, b, t = smpl_dict['pose'], smpl_dict['betas'], smpl_dict['trans']
pose = torch.tensor(p[None, ...])
betas = torch.tensor(b[None, ...])
trans = torch.tensor(t[None, ...])


verts, jtr,_,  _ = smpl(pose, th_betas=betas, th_trans=trans)
verts = verts[0].cpu().numpy()
faces = smpl.th_faces.cpu().numpy()
jtr = jtr.cpu().numpy()


smpl_vert_segmentation = json.load(open('support_data/smpl_vert_segmentation.json', 'r'))

smpl_vert_face_segmentation = {}

vert_to_partname = ['' for _ in range(verts.shape[0])]

for name in smpl_vert_segmentation:
    vertice_list = smpl_vert_segmentation[name]

    for vertice_id in vertice_list:
        vert_to_partname[vertice_id] = name

# print(vert_to_partname)

face_to_partname = []

for i in range(faces.shape[0]):
    names = [vert_to_partname[vid] for vid in faces[i]]


    if not (names[0] == names[1]) & (names[1] == names[2]):
        for n in names:
            if names.count(n) == 2:
                name = n
                break

    else:
        name = names[0]

    face_to_partname.append(name)
    # print(name1, name2, name3)


assert len(vert_to_partname) == verts.shape[0]
assert len(face_to_partname) == faces.shape[0]

res = {
    'vert_to_parts': vert_to_partname,
    'face_to_parts': face_to_partname,
    'faces': faces.tolist()
}

with open('support_data/smpl_vertface_parts.json', 'w') as f:

    json.dump(res, f)