import matplotlib.pyplot as plt
import numpy as np
import tempfile
import mxnet as mx
import sagemaker
from sagemaker.amazon.record_pb2 import Record
import random 

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

class SSProtobufDeserializer(sagemaker.deserializers.BaseDeserializer):
    """Deserialize protobuf semantic segmentation response into a numpy array"""

    def __init__(self, accept="application/x-protobuf"):
        self.accept = accept

    @property
    def ACCEPT(self):
        return (self.accept,)

    def deserialize(self, stream, content_type):
        """Read a stream of bytes returned from an inference endpoint.
        Args:
            stream (botocore.response.StreamingBody): A stream of bytes.
            content_type (str): The MIME type of the data.
        Returns:
            mask: The numpy array of class confidences per pixel
        """
        try:
            rec = Record()
            # mxnet.recordio can only read from files, not in-memory file-like objects, so we buffer the
            # response stream to a file on disk and then read it back:
            with tempfile.NamedTemporaryFile(mode="w+b") as ftemp:
                ftemp.write(stream.read())
                ftemp.seek(0)
                recordio = mx.recordio.MXRecordIO(ftemp.name, "r")
                protobuf = rec.ParseFromString(recordio.read())
            values = list(rec.features["target"].float32_tensor.values)
            shape = list(rec.features["shape"].int32_tensor.values)
            # We 'squeeze' away extra dimensions introduced by the fact that the model can operate on batches
            # of images at a time:
            shape = np.squeeze(shape)
            mask = np.reshape(np.array(values), shape)
            return np.squeeze(mask, axis=0)
        finally:
            stream.close()


def random_tuple(x, y, iteranons= 5):
    
    pixel_list = []
    for i in range(iteranons):
        pixel = [random.randint(0, x- 50), random.randint(0, y-50)]
        pixel_list.append(pixel)
        
    return pixel_list
