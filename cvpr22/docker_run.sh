docker build --tag cvpr:1.0 .

docker run -d -it --name cvpr --gpus all -v .:/data --restart=always cvpr:1.0 

docker exec -it cvpr /bin/bash