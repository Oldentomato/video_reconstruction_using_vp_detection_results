docker run -d -it --shm-size=8G --name neur --gpus all -v .:/data --restart=always vpenv:1.0 

docker exec -it neur /bin/bash