import sys
sys.path.append("..")

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from datasets.backdoor_dataset import CIFAR10M,CustomDataset_224,CIFAR10Mem
import numpy as np
from datasets.bd_dataset_imagenet_filter import BadEncoderDataset


def cifar10_dataloader(args):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    clean_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])

    memory_data = CIFAR10M(numpy_file='../data/cifar10/train.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    train_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_data = CIFAR10M(numpy_file='../data/cifar10/test.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader,test_loader


def stl10_dataloader(args):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
        ])

    clean_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
                ])


    memory_data = CIFAR10M(numpy_file='../data/stl10/train_unlabeled.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    train_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_data = CIFAR10M(numpy_file='../data/stl10/test.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)


    return train_loader,test_loader


def gtsrb_dataloader(args):
    classes = ['Speed limit 20km/h',
                            'Speed limit 30km/h',
                            'Speed limit 50km/h',
                            'Speed limit 60km/h',
                            'Speed limit 70km/h',
                            'Speed limit 80km/h', #5
                            'End of speed limit 80km/h',
                            'Speed limit 100km/h',
                            'Speed limit 120km/h',
                            'No passing sign',
                            'No passing for vehicles over 3.5 metric tons', #10
                            'Right-of-way at the next intersection',
                            'Priority road sign',
                            'Yield sign',
                            'Stop sign', #14
                            'No vehicles sign',  #15
                            'Vehicles over 3.5 metric tons prohibited',
                            'No entry',
                            'General caution',
                            'Dangerous curve to the left',
                            'Dangerous curve to the right', #20
                            'Double curve',
                            'Bumpy road',
                            'Slippery road',
                            'Road narrows on the right',
                            'Road work',    #25
                            'Traffic signals',
                            'Pedestrians crossing',
                            'Children crossing',
                            'Bicycles crossing',
                            'Beware of ice or snow',   #30
                            'Wild animals crossing',
                            'End of all speed and passing limits',
                            'Turn right ahead',
                            'Turn left ahead',
                            'Ahead only',   #35
                            'Go straight or right',
                            'Go straight or left',
                            'Keep right',
                            'Keep left',
                            'Roundabout mandatory', #40
                            'End of no passing',
                            'End of no passing by vehicles over 3.5 metric tons']

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.3389, 0.3117, 0.3204], [0.2708, 0.2588, 0.2618])
        ])

    clean_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.3389, 0.3117, 0.3204], [0.2708, 0.2588, 0.2618])
                ])


    memory_data = CIFAR10M(numpy_file='../data/gtsrb/train.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    train_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_data = CIFAR10M(numpy_file='../data/gtsrb/test.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)


    return train_loader,test_loader

def svhn_dataloader(args):
    classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize([0.3389, 0.3117, 0.3204], [0.2708, 0.2588, 0.2618])
        ])

    clean_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.3389, 0.3117, 0.3204], [0.2708, 0.2588, 0.2618])
                ])


    memory_data = CIFAR10M(numpy_file='../data/gtsrb/train.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    train_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_data = CIFAR10M(numpy_file='../data/gtsrb/test.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)


    return train_loader,test_loader

def imagenet_dataloader(args):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.CenterCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    clean_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classes = [str(i) for i in range(1000)]


    training_data_num = 1000000
    np.random.seed(3047)
    training_data_sampling_indices = np.random.choice(training_data_num, int(training_data_num*0.01), replace=False)

    shadow_dataset = BadEncoderDataset(
        root = "../data/imagenet/train",
        class_type=classes,indices = training_data_sampling_indices,
        transform=clean_transform,
        bd_transform=train_transform,
    )

    # shadow_dataset = BadEncoderDataset(
    #     root = "../data/imagenet/train",
    #     class_type=classes,indices = training_data_sampling_indices,
    #     transform=clean_transform,
    #     bd_transform=train_transform,
    # )

    train_loader = DataLoader(shadow_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return train_loader


def imagenet_all_dataloader(args):
    class ConvertToRGB:
        def __call__(self, image):
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.CenterCrop(size=(224, 224)),
        ConvertToRGB(), # 将单通道的转换为3通道的
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # train_transform2 = transforms.Compose([
    #     ConvertToRGB(),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor(),
    #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])

    clean_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classes = [str(i) for i in range(1000)]


    training_data_num = 1000000
    np.random.seed(3047)
    training_data_sampling_indices = np.random.choice(training_data_num, int(training_data_num*0.01), replace=False)

    imagenet_dataset = BadEncoderDataset(
        root = "../data/imagenet/train",
        class_type=classes,indices = training_data_sampling_indices,
        transform=clean_transform,
        bd_transform=train_transform,
    )


    gtsrb_data = CustomDataset_224(directory='../data/gtsrb/selected_train_224', transform1=clean_transform,transform2=train_transform)
    stl10_data = CustomDataset_224(directory='../data/stl10/selected_train_224', transform1=clean_transform,transform2=train_transform)
    svhn_data = CustomDataset_224(directory='../data/svhn/selected_train_224', transform1=clean_transform,transform2=train_transform)

    train_dataset = ConcatDataset([imagenet_dataset, gtsrb_data,stl10_data,svhn_data])

    # gtsrb_data = CustomDataset_224(directory='../data/gtsrb/train_224', transform1=clean_transform,transform2=train_transform)
    # train_dataset = ConcatDataset([imagenet_dataset, gtsrb_data])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return train_loader
