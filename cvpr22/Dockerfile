FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
COPY . /data
WORKDIR /data

RUN apt-get update && \
    python -m pip install --upgrade pip && \
    apt-get -y install libgl1-mesa-glx && \
    pip install -r requirements.txt

CMD pip install torch_scatter


