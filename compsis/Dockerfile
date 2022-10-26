FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime 

ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

## Pyton installation ##
RUN apt-get update 
RUN apt-get install -y apt-utils
RUN apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y
RUN pip3 install opencv-contrib-python
RUN apt-get install -y zip screen
RUN pip3 install lxml tqdm absl-py easydict Pillow numpy scipy
RUN pip3 install waitress

RUN pip3 install flask flask_jwt_extended pymongo
RUN pip3 install seaborn thop

RUN mkdir /app

WORKDIR /app

CMD ["python3","-u","./local/app.py"]