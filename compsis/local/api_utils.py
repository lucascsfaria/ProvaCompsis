import cv2
import torch
import ast

from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device


def load_model(weights):
    # Inicializa modelo
    device = select_device("0")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(640, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    print('\n>>>>>>>>>> O modelo foi carregado com sucesso!\n')

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    return model, stride, imgsz, device, half

def read_request(data_json):

    with open(data_json["filename_txt"]) as f:
        lines = f.readlines()
    
    list_frames =[]
    for line in lines:
        if(line!='\n'):
            line_split = line.split('|')
            frame = {
                "idx": int(line_split[0]),
                "infer": ast.literal_eval(line_split[1]),
                "cam": line_split[2],
                "type_cam": line_split[3],
                "time":line_split[4]
            }
            list_frames.append(frame)
    return list_frames

def image_from_video(data_json):
    list_images = []
    vidcap = cv2.VideoCapture(data_json["filename_mp4"])
    success,image = vidcap.read()
    count = 0
    while success:
        list_images.append(image)#converte para RGB
        #cv2.imwrite("input/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
        
    return list_images


def prepare_return(response):
    response = response.replace("b'", "'")
    response = response.replace("'", '"')
    response = response.replace(")", "")
    response = response.replace("(", "")

    return response

