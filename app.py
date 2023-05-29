import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

from flask import Flask, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

import json

import matplotlib.pyplot as plt
import cv2

from io import BytesIO
import base64

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys

from utlis import *

sys.path.append("..")

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

@app.route('/sam', methods=['POST'])
def upload():
    
    data = request.get_json()
    image_base64 = data.get('image')
    image_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    masks = mask_generator.generate(image)

    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, format='jpg')
    buffer.seek(0)

    # Encode the image as base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    data = {'code': 200,
            'message': "The material was detected sucessfully",
            'image': image_base64}
                    
    r = Response(response= json.dumps(data), status=200, mimetype="application/json")
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Methods"] = "OPTIONS,POST,GET"
    # Print the base64 encoded image
    return r

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
