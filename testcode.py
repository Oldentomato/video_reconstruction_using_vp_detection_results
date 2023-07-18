import numpy as np


with open("etri_cart_200219_15h01m_2fps_gt3.txt","r") as f:
    ground_t = f.read().splitlines()

print((int(float(ground_t[0].split(",")[2])),int(float(ground_t[0].split(",")[3]))))