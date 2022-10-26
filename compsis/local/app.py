import os
import json
import torch
import traceback
import numpy as np
import torch.backends.cudnn as cudnn

from waitress import serve
from flask import Flask, request, jsonify
from utils.general import non_max_suppression, scale_coords
from api_utils import load_model, read_request, image_from_video, prepare_return

app = Flask(__name__)

@app.route('/repescagem', methods=['POST'])
def predict():
    try:
        if request.headers.get('content-type') == "application/json":
            data_json = request.get_json()

            list_frames = read_request(data_json)
  
            list_images = image_from_video(data_json)


            for frame in list_frames:

                frame["valid_infer"] = []
                for detection in frame['infer']:

                    #Se existir inferencia da classe carro, executa inferencia no modelo
                    if str(detection[0]) == "b'car'":

                        #Processa detecção do arquivo .txt
                        x, y, w, h= detection[2]
                        image = list_images[frame["idx"]]
                        image_cropped = image[int(y):int(y+h), int(x):int(x+w)] 

                        #Preparando imagem com corte do objeto para inferencia no modelo

                        img_inference = np.zeros([640,640,3],dtype=np.uint8)
                        img_inference[0:image_cropped.shape[0], 0:image_cropped.shape[1]] = image_cropped
                        #cv2.imwrite("input/frame%d.jpg" % count, image_cropped)     # save frame as JPEG file
                        img_inference = img_inference[:, :, ::-1].transpose(2, 0, 1)
                        img_inference = np.ascontiguousarray(img_inference)
                        img_inference = torch.from_numpy(img_inference).to(device)
                        img_inference = img_inference.half() if half else img_inference.float()  # uint8 to fp16/32
                        img_inference /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img_inference.ndimension() == 3:
                            img_inference = img_inference.unsqueeze(0)
                        
                        #Evita vazamento de mémoria na GPU
                        with torch.no_grad():
                            pred = model(img_inference)[0]
                        
                        pred = non_max_suppression(pred, 0.25, 0.45)
                        
                        # Processa todas detecções da imagem criada
                        detections_classes = []

                        for i, det in enumerate(pred):
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img_inference.shape[2:], det[:, :4], img_inference.shape).round()
                                
                                # Write results
                                for *xyxy, conf, cls in reversed(det):
                                    detections_classes.append(names[int(cls)])
                                    label = f'{names[int(cls)]} {conf:.2f}'
                                    print("Classe: ", label)
                                    print("BBOX: ", (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])))

                        if 'car' in detections_classes:
                            frame["valid_infer"].append(detection)
            
            
            #Prepara a resposta em formato JSON, removendo caracteres não suportados
            response = prepare_return(str(list_frames))
            
            return jsonify(json.loads(response))
        else:
            return jsonify({"error": "Request no formato incorreto"})
    except:
        return jsonify({'trace': traceback.format_exc()})

@app.route('/', methods=['GET'])
def online():
    return jsonify({"response": "Aplicacao Online"})


if __name__ == '__main__':
    weigths = os.environ['MODEL_PATH']

    #Carrega modelo ao iniciar aplicação para não ter que ficar carregando em todas inferencias
    model, stride, imgsz, device, half = load_model(weigths)
    names = model.module.names if hasattr(model, 'module') else model.names

    serve(app, host="0.0.0.0", port=5000)
