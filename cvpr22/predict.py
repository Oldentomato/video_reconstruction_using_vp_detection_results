
#!/usr/bin/env python3
"""Compute vanishing points from images.
Usage:
    demo.py [options] <yaml-config> <checkpoint> <image>
    demo.py ( -h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint

Options:
   -h --help                     Show this screen
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   -o --output <output>          Path to the output AA curve [default: error.npz]
   --dump <output-dir>           Optionally, save the vanishing points to npz format.
"""

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
import math

class Detect_VP:

    def topk_orthogonal_vps(self, scores, xyz, num_vps=3):

        index = np.argsort(-scores)
        vps_idx = [index[0]]
        for i in index[1:]:
            if len(vps_idx) == num_vps:
                break
            # cos_distance function: input: x: mxp, y: nxp; output: y, mxn
            ### scipy: same 0, opposite 2, orthorgonal 1, dist = 1-AB/(|A||B|)
            dist_cos = scipy_spatial_dist.cdist(xyz[vps_idx], xyz[i][None, :], 'cosine')
            ### same 1, opposite -1, orthorgonal 0
            dist_cos = np.abs(-1.0*dist_cos+1.0)

            dist_cos_arc = np.min(np.arccos(dist_cos))
            if dist_cos_arc >= np.pi/num_vps:
                vps_idx.append(i)
            else:
                continue

        vps_pd = xyz[vps_idx]
        return vps_pd, vps_idx
    


    def to_pixel(self, vpts, focal_length=1.0, h=480, w=640):
        x = vpts[:,0] / vpts[:, 2] * focal_length * max(h, w)/2.0 + w//2
        y = -vpts[:,1] / vpts[:, 2] * focal_length * max(h, w)/2.0 + h//2
        return y, x


    def _AA(self, a, b, thresh):
        v = 0
        k = [i for i, val in enumerate(a) if val < thresh]

        if not k:
            return v

        a = [a[i] for i in k]
        a.append(thresh)

        n = len(a)
        for i in range(n - 1):
            v = v + (a[i + 1] - a[i]) * b[i]

        return v

    def __init__(self,model_path,yaml_name):

        # file = open("etri_cart_200219_15h01m_2fps_gt3.txt","r")
        # self.ground_t = file.readlines()
        # file.close()

        self.frame_count = 0

        self.model_path = model_path

        C.update(C.from_yaml(filename=f"./config/{yaml_name}.yaml"))
        C.model.im2col_step = 32  # override im2col_step for evaluation
        M.update(C.model)
        pprint.pprint(C, indent=4)

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        

        device_name = "gpu"
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
        self.device = torch.device(device_name)

        npzfile = np.load(C.io.ht_mapping, allow_pickle=True)
        ht_mapping = npzfile['ht_mapping']
        ht_mapping[:,2] = npzfile['rho_res'].item() - np.abs(ht_mapping[:,2])
        ht_mapping[:,2] /= npzfile['rho_res'].item()
        vote_ht_dict={}
        vote_ht_dict["vote_mapping"]= torch.tensor(ht_mapping, requires_grad=False).float().contiguous()
        vote_ht_dict["im_size"]= (npzfile['rows'], npzfile['cols'])
        vote_ht_dict["ht_size"]= (npzfile['h'], npzfile['w'])


        npzfile = np.load(C.io.sphere_mapping, allow_pickle=True)
        sphere_neighbors = npzfile['sphere_neighbors_weight'] #nyu: sphere_neighbors_weight
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

        self.model = vpd.models.VanishingNet(backbone, vote_ht_dict, vote_sphere_dict)
        self.model = self.model.to(self.device)
        self.model = torch.nn.DataParallel(
            self.model, device_ids=list(range(1))
        )


        checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


        ##### number of parameters in a model
        total_params = sum(p.numel() for p in self.model.parameters())
        ##### number of trainable parameters in a model
        train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('num of total parameters', total_params)
        print('num of trainable parameters', train_params)

        self.xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=C.io.num_nodes)


    
    def _vector_magnitude(self, x, y):
        # 벡터의 크기(절대값)를 계산합니다.
        return math.sqrt(x**2 + y**2)

    def dot_product_cosine_angle(self, x1, y1, x2, y2):
        # 두 점의 벡터를 구합니다.
        vector1 = (x1, y1)
        vector2 = (x2, y2)

        # 두 벡터의 크기를 계산합니다.
        magnitude1 = self._vector_magnitude(vector1[0], vector1[1])
        magnitude2 = self._vector_magnitude(vector2[0], vector2[1])

        # 두 벡터의 내적(dot product)을 계산합니다.
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        # 두 벡터의 코사인 각도를 계산합니다.
        cosine_angle = dot_product / (magnitude1 * magnitude2)

        # 코사인 각도를 라디안으로 변환하여 반환합니다.
        angle_in_radians = math.acos(cosine_angle)

        # 라디안을 도(degree)로 변환하여 반환합니다.
        angle_in_degrees = math.degrees(angle_in_radians)

        return angle_in_degrees


    def predict(self, input_img, img_size):
        image = input_img
        origin_image = input_img

        if image.shape[0:2]!=tuple([img_size[1], img_size[0]]):
            image = cv2.resize(image, dsize=(img_size[0], img_size[1]))
            image *= 255
        image = np.rollaxis(image, 2).copy()
        image = torch.from_numpy(image).float().to(self.device)
        targets = torch.zeros(C.io.num_nodes).float().to(self.device)
        input_dict = {"image": image[None],  "target": targets, "eval": True}

        with torch.no_grad():
            result = self.model(input_dict)
        pred = result["prediction"].cpu().numpy()[0]
        
        # Option 1:
        # a. f available: first map to camera space, and then pick up the top3;
        # b. Assumption: VPs are more or less equally spread over the sphere.
        vpts_pd, vpts_idx = self.topk_orthogonal_vps(pred, self.xyz, num_vps=3)
        
        
        ys, xs = self.to_pixel(vpts_pd, focal_length=1.0, h=720, w=1280)


        result_x = -1
        result_y = -1


        resultcoord = []
        result_count = 0
        

        ### visualize results on the hemisphere
        for (x, y) in zip(xs, ys):
            if (x <= 1280 and x >= 0) and (y <= 720 and y >= 0):
                result_x = x
                result_y = y
                resultcoord.insert(0,(x,y))
                result_count += 1
            else:
                resultcoord.append((x,y))

        origin_image = cv2.circle(origin_image, (int(result_x),int(result_y)), radius=20, color=(0,255,0), thickness=-1)

        #조건 정렬
        sorted(resultcoord, key=lambda x: (x[0], x[1])if (x[0], x[1]) == (result_x,result_y) else x)
        return (result_x,result_y,result_count,origin_image,resultcoord)











