import torch
import torchvision

from flask import Flask, request, jsonify
app = Flask(__name__)

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
device = "cpu"

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

    # Print the base64 encoded image
    return jsonify({'image_base64': image_base64})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
