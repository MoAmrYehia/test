import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from io import BytesIO
import base64

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import sys
sys.path.append("..")

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread('images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
print(image_base64)
