from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer 
import json

model_root = "support_data/mano"

  
smpl = SMPL_Layer(center_idx=0, gender="male", num_betas=10,
                    model_root=str(model_root), hands=True)


with open('support_data/smplh_kintree_parents.json', 'w') as f:
    json.dump(smpl.kintree_parents, f)