# Start
## Docker
### initial
docker run -itd --name=apisumm --shm-size 16G --gpus all -it --mount 'type=bind,src=/storage/chengran/APISummarization,dst=/workspace' minimal_toolkit:1.1
### start
docker exec -it bb8d8bffaf12 /bin/bash
