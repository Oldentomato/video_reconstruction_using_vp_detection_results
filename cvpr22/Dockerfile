FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
COPY requirements.txt /data/requirements.txt
WORKDIR /data

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-get update 

RUN python -m pip install --upgrade pip && \
    apt-get -y install libglib2.0-0 && \
    apt-get -y install libgl1-mesa-glx && \
    pip install -r requirements.txt && \
    pip install torch-scatter
    # pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.0+cu113.html && \
    # pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.0+cu113.html



