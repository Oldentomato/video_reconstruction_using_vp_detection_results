import numpy as np
import math



with open("etri_cart_200219_15h01m_2fps_gt3.txt","r") as f:
    ground_t = f.readlines()
with open("saved_results/ScanNet/result.txt","r") as f:
    pred_t = f.readlines()


