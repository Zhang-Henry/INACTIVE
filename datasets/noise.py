from io import BytesIO
import torch
import numpy as np
import random
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms

def add_salt_and_pepper_noise(image, probability=0.05):
    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)
    output = np.array(image)
    thres = 1 - probability
    for i in range(image.size[1]):
        for j in range(image.size[0]):
            rand = random.random()
            if rand < probability:
                output[i][j] = 0       # black pixel
            elif rand > thres:
                output[i][j] = 255     # white pixel
    return output


def randomJPEGcompression(image):
    qf = random.randrange(10, 100)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

# def JPEGcompression(image, quality=1):
#     outputIoStream = BytesIO()
#     image.save(outputIoStream, "JPEG", quality=quality, optimize=True)
#     outputIoStream.seek(0)
#     return Image.open(outputIoStream)



def add_poisson_noise(image, scale=1.0):
    """
    Add Poisson noise to the image.
    The `scale` parameter reduces the noise if it's less than 1, increases otherwise.
    """
    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)

    image_np = np.array(image)
    # Scale down the image to reduce the noise
    scaled_image = image_np / scale
    # Generate Poisson noise
    noise = np.random.poisson(scaled_image).astype(np.float32)
    # Scale up the noisy image to maintain the original range
    noisy_image = (noise * scale).clip(0, 255)

    return noisy_image.astype(np.uint8)


def add_quantization_noise(image, levels=64):
    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)
    output = np.array(image)
    max_val = 255
    quantized = np.floor(output / (max_val / levels)) * (max_val / levels)
    return quantized


# def bit_depth_red(X_before,depth):
#     r=256/(2**depth)
#     x_quan=torch.round(X_before*255/r)*r/255
#     return x_quan

# def JPEGcompression(X_before,quality):
#         X_after=torch.zeros_like(X_before)
#         for j in range(X_after.size(0)):
#             x_np=transforms.ToPILImage()(X_before[j].detach().cpu())
#             x_np.save('./'+'j.jpg',quality=quality)
#             X_after[j]=Image.open('./'+'j.jpg')
#         return X_after
