FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
COPY requirements.txt /data/requirements.txt
WORKDIR /data

RUN apt-get update && \
    python -m pip install --upgrade pip && \
    apt-get -y install libglib2.0-0 && \
    apt-get -y install libgl1-mesa-glx && \
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.0+cu113.html && \
    pip install -r requirements.txt



