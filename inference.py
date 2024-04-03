import base64
import gzip
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabHead


class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        self.model.classifier = DeepLabHead(960, 1)

    def forward(self, input_data):
        return self.model(input_data)


def model_fn(model_dir):
    model = CustomModel()
    with open(os.path.join(model_dir, 'model.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/octet-stream':
        compressed_img_io = io.BytesIO(request_body)
        decompressed_img_io = gzip.GzipFile(fileobj=compressed_img_io, mode='rb')

        image_bytes = io.BytesIO(decompressed_img_io.read())

        img = Image.open(image_bytes)
        image = img.convert("RGB")
        image = transforms.ToTensor(image)
        img_tensor = image.unsqueeze(0) if len(image.shape) == 3 else image
        return img_tensor
    else:
        raise ValueError(
            f"Request content type {request_content_type} is not supported. Please send 'application/octet-stream' "
            f"and pass the byte array of the image.")


def predict_fn(input_data, model):
    with torch.no_grad():
        activation = torch.nn.Sigmoid()
        threshold = np.loadtxt("logs/best_threshold.txt")

        mask = model(input_data)["out"]
        mask = activation(mask)

        image_numpy = input_data.numpy()[0].transpose(1, 2, 0)
        mask_numpy = mask[0, 0].detach().numpy()

        indices = np.where(mask_numpy > threshold)
        tmp = mask_numpy[indices]

        damaged_pixels = np.count_nonzero(mask_numpy > threshold)
        title = "Found %d Damaged Pixels" % damaged_pixels

        mask_numpy[:, :] = np.nan
        mask_numpy[indices] = tmp

        fig = plt.figure()
        title, image_numpy, mask_numpy, alpha = "Damaged Pixels", image_numpy, mask_numpy, 0.2
        fig, ax = plot_single(fig, title, image_numpy, mask_numpy, alpha)
        plt.show()
        plt.close()

        with open('output.png', "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return {'plot': encoded_string}


def output_fn(prediction_output, accept):
    if accept == 'application/json':
        return prediction_output, 'application/json'
    raise Exception("Requested unsupported ContentType in accept: " + accept)


def plot_single(fig, title, input_image, output_mask, alpha=0.2):
    ax = fig.add_subplot(111)
    ax.imshow(input_image, cmap="gray", interpolation="None")
    m = ax.imshow(output_mask, cmap="jet", interpolation="None", vmin=0, vmax=1, alpha=alpha)
    ax.set_title(title)
    fig.colorbar(m)
    ax.axis("off")
    return fig, ax