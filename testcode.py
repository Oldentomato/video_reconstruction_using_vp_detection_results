
import os
import pprint
import random
import torch
import numpy as np
import scipy.spatial.distance as scipy_spatial_dist
import vpd
import cv2
from vpd.config import C, M
from vpd.models.sphere.sphere_utils import gold_spiral_sampling_patch

C.update(C.from_yaml(filename=f"config/nyu.yaml"))
C.model.im2col_step = 32  # override im2col_step for evaluation
M.update(C.model)
pprint.pprint(C, indent=4)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


device_name = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device_name = "cuda"
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(0)
    print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    for k in range(0, torch.cuda.device_count()):
        print('kth, device name', k, torch.cuda.get_device_name(k))
else:
    print("CUDA is not available")

device = torch.device(device_name)

npzfile = np.load(C.io.ht_mapping, allow_pickle=True)
ht_mapping = npzfile['ht_mapping']
ht_mapping[:,2] = npzfile['rho_res'].item() - np.abs(ht_mapping[:,2])
ht_mapping[:,2] /= npzfile['rho_res'].item()
vote_ht_dict={}
vote_ht_dict["vote_mapping"]= torch.tensor(ht_mapping, requires_grad=False).float().contiguous()
vote_ht_dict["im_size"]= (npzfile['rows'], npzfile['cols'])
vote_ht_dict["ht_size"]= (npzfile['h'], npzfile['w'])


npzfile = np.load(C.io.sphere_mapping, allow_pickle=True)
sphere_neighbors = npzfile['sphere_neighbors_weight']
vote_sphere_dict={}
vote_sphere_dict["vote_mapping"]=torch.tensor(sphere_neighbors, requires_grad=False).float().contiguous()
vote_sphere_dict["ht_size"]=(npzfile['h'], npzfile['w'])
vote_sphere_dict["sphere_size"]=npzfile['num_points']



# 2. model
if M.backbone == "stacked_hourglass":
    backbone = vpd.models.hg(
        planes=128, depth=M.depth, num_stacks=M.num_stacks, num_blocks=M.num_blocks
    )
else:
    raise NotImplementedError

model = vpd.models.VanishingNet(backbone, vote_ht_dict, vote_sphere_dict)

print(model)