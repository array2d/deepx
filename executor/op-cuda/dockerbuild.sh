cd ../
pwd
ls -al
docker build -t docker.array2d.com/deepx/cuda:latest . -f op-mem-cuda/Dockerfile
#docker push docker.array2d.com/deepx/cuda:latest
