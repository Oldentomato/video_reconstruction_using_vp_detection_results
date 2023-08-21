## Video reconstruction using vp detection results  
### Used sources
- [(cvpr'22)VanishingPoint_HoughTransform_GaussianSphere](https://github.com/yanconglin/VanishingPoint_HoughTransform_GaussianSphere) (Dockerfile Ready)
- [(neurVPS)NeurVPS](https://github.com/zhou13/neurvps)  

### Specification  
|name|info|
|---|---|
|os|ubuntu18.04|
|cpu|Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz|
|gpu|GeForce RTX 2080 Ti Rev. A|
|ram|16G|


### Libraries 
|name|version|
|---|---|
|python|3.7.0|
|pytorch|1.12.0-cuda11.3-cudnn8|
|numpy|1.21.2|
|matplotlib|3.2.2|
|opencv-python|4.8.0.74|
|scipy|1.7.3|
|scikit-image|0.18.3|

### Metrics  
- Performance  
    |name|sec per frame(avg)|
    |---|---|
    |NYU(cvpr)|0.20 sec|
    |ScanNet(cvpr)|0.17 sec|
    |SU3(cvpr)|0.18 sec|
    |SU3(neur)|1.09 sec|
    |ScanNet(neur)|1.20 sec|
    |TMM17(neur)|0.85 sec|
- AA(Angular Accuracy) Graph  
    - cvpr  
    ![img1](https://github.com/Oldentomato/detect_vp-reconstruction_vid/blob/main/README_imgs/AA_graph_cvpr.png)  
    - neur  
    ![img2](https://github.com/Oldentomato/detect_vp-reconstruction_vid/blob/main/README_imgs/AA_graph_neur.png)  
   - all  
    ![img3](https://github.com/Oldentomato/detect_vp-reconstruction_vid/blob/main/README_imgs/AA_graph.png)  

 ### SnapShot  
 - cvpr NYU Dataset Result  
   

### Reference paper  
[(cvpr) https://arxiv.org/abs/2203.08586](https://arxiv.org/abs/2203.08586)  
[(neurVPS) https://arxiv.org/abs/1910.06316](https://arxiv.org/abs/1910.06316)