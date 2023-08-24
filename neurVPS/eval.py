import os
import math
import random
import numpy as np
import torch,gc
import numpy.linalg as LA
import cv2
import neurvps
from neurvps.config import C, M

class Detect_VP:

    def __init__(self, model_dir, config_name):
        gc.collect()
        torch.cuda.empty_cache()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        C.update(C.from_yaml(filename=f"./config/{config_name}.yaml"))
        # C.model.im2col_step = 32  # override im2col_step for evaluation
        M.update(C.model)

        device_name = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if torch.cuda.is_available():
            device_name = "cuda"
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(0)
            print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        else:
            print("CUDA is not available")
        device = torch.device(device_name)

        self.device = device

        if M.backbone == "stacked_hourglass":
            model = neurvps.models.hg(
                planes=64, depth=M.depth, num_stacks=M.num_stacks, num_blocks=M.num_blocks
            )
        else:
            raise NotImplementedError


        checkpoint = torch.load(model_dir)
        model = neurvps.models.VanishingNet(
            model, C.model.output_stride, C.model.upsample_scale
        )
        model = model.to(device)
        model = torch.nn.DataParallel(
            model, device_ids=list(range(1))
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()


        self.model = model

    
    def to_pixel(self,w):
        x = w[0] / w[2] * C.io.focal_length * max(720,1280)/2.0 + 1280//2
        y = -w[1] / w[2] * C.io.focal_length * max(720,1280)/2.0 + 720//2
        return y, x



    def predict(self, image, size):
        n = 3 #top 3 vp

        origin_img = image
        # image = image.astype(float)[:,:,:3]
        image = cv2.resize(image, dsize=size)
        # image *= 255
        image = np.rollaxis(image, 2).copy()
        image = torch.from_numpy(image).float().to(self.device)
        image = image.unsqueeze(0)
        input_dict = {"image":image, "test": True}
        #node_num (sample_sphere마지막 인자값)이 샘플링 좌표들인데 이 좌표가 많으면 많을수록 더 정교하게 좌표가
        #지정이 된다. 하지만 지금은 cuda memory error로 인해 256까지 밖에 안된다.
        vpts = self.sample_sphere(np.array([0, 0, 1]), 90.0 * np.pi / 180., 64)
        input_dict["vpts"] = torch.tensor(vpts)
        with torch.no_grad():
            score = self.model(input_dict)[:, -1].cpu().numpy()

        index = np.argsort(-score)

        candidate = [index[0]]
        for i in index[1:]:
            if len(candidate) == n:
                break
            dst = np.min(np.arccos(np.abs(vpts[candidate] @ vpts[i])))

            if dst >= np.pi/n:
                candidate.append(i)
            else:
                continue
        vpts_pd = vpts[candidate]

        for res in range(1, len(M.multires)):
            vpts = [self.sample_sphere(vpts_pd[vp], M.multires[-res], 64) for vp in range(n)]
            input_dict["vpts"] = np.vstack(vpts)
            with torch.no_grad():
                score = self.model(input_dict)[:, -res - 1].cpu().numpy().reshape(n, -1)
            for i, s in enumerate(score):
                vpts_pd[i] = vpts[i][np.argmax(s)]



        result_x = -1
        result_y = -1
        resultcoord = []
        result_count = 0
        for coord in vpts_pd:
            y,x = self.to_pixel(coord)
            if x <= 1280 and y <= 720 and x >= 0 and y >= 0:
                resultcoord.insert(0,(x,y))
                result_x = x
                result_y = y
                result_count += 1
            else:
                resultcoord.append((-1,-1))


        # origin_img = cv2.circle(origin_img, (int(result_x),int(result_y)), radius=20, color=(255,0,0), thickness=-1)
        sorted(resultcoord, key=lambda x: (x[0], x[1])if (x[0], x[1]) == (result_x,result_y) else x)
        return (result_x,result_y,result_count,origin_img,resultcoord)



    def sample_sphere(self, v, alpha, num_pts):
        v1 = self.orth(v)
        v2 = np.cross(v, v1)
        v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
        indices = np.linspace(1, num_pts, num_pts)
        phi = np.arccos(1 + (math.cos(alpha) - 1) * indices / num_pts)
        theta = np.pi * (1 + 5 ** 0.5) * indices
        r = np.sin(phi)
        return (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T


    def orth(self, v):
        x, y, z = v
        o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
        o /= LA.norm(o)
        return o

