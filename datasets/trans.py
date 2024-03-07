from torchvision import transforms
from .noise import *


test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10_GaussianBlur = transforms.Compose([
    transforms.GaussianBlur(kernel_size=7),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10_JPEGcompression = transforms.Compose([
    lambda x: JPEGcompression(x, quality=20),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10_salt_and_pepper_noise = transforms.Compose([
    lambda x: add_salt_and_pepper_noise(x, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


test_transform_cifar10_poisson_noise = transforms.Compose([
    lambda x: add_poisson_noise(x, scale=10),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


test_transform_stl10_GaussianBlur = transforms.Compose([
    transforms.GaussianBlur(kernel_size=7),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_stl10_JPEGcompression = transforms.Compose([
    lambda x: JPEGcompression(x, quality=2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_stl10_salt_and_pepper_noise = transforms.Compose([
    lambda x: add_salt_and_pepper_noise(x, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


test_transform_stl10_poisson_noise = transforms.Compose([
    lambda x: add_poisson_noise(x, scale=5),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    ])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])