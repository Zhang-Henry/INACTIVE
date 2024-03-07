from torchvision import transforms
from .backdoor_dataset import *
import numpy as np
from .trans import *



classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']




def get_downstream_svhn(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        if args.noise == 'GaussianBlur':
            test_transform = test_transform_cifar10_GaussianBlur
            print('test_transform_cifar10_GaussianBlur')
        elif args.noise == 'JPEGcompression':
            test_transform = test_transform_cifar10_JPEGcompression
            print('test_transform_cifar10_JPEGcompression')
        elif args.noise == 'salt_and_pepper_noise':
            test_transform = test_transform_cifar10_salt_and_pepper_noise
            print('test_transform_cifar10_salt_and_pepper_noise')
        elif args.noise == 'poisson_noise':
            test_transform = test_transform_cifar10_poisson_noise
            print('test_transform_cifar10_poisson_noise')
        else:
            test_transform = test_transform_cifar10
            print('test_transform_cifar10')
        memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    elif args.encoder_usage_info == 'stl10':
        if args.noise == 'GaussianBlur':
            test_transform = test_transform_stl10_GaussianBlur
            print('test_transform_stl10_GaussianBlur')
        elif args.noise == 'JPEGcompression':
            test_transform = test_transform_stl10_JPEGcompression
            print('test_transform_stl10_JPEGcompression')
        elif args.noise == 'salt_and_pepper_noise':
            test_transform = test_transform_stl10_salt_and_pepper_noise
            print('test_transform_stl10_salt_and_pepper_noise')
        elif args.noise == 'poisson_noise':
            test_transform = test_transform_stl10_poisson_noise
            print('test_transform_stl10_poisson_noise')
        else:
            test_transform = test_transform_stl10
            print('test_transform_stl10')
        memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224'
        testing_file_name = 'test_224'
        memory_data = CIFAR10Mem_224(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        test_data_backdoor = BadEncoderTestBackdoor_224(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        test_data_clean = CIFAR10Mem_224(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)


    return target_dataset, memory_data, test_data_clean, test_data_backdoor
