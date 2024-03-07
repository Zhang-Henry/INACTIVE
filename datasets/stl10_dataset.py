from torchvision import transforms
from .backdoor_dataset import *
import numpy as np
from .trans import *

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

finetune_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])




backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])



classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


def get_pretraining_stl10(data_dir):
    train_data = CIFAR10Pair(numpy_file=data_dir + "train_unlabeled.npz", class_type= classes, transform=train_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type= classes, transform=test_transform_stl10)
    test_data  = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type= classes,transform=test_transform_stl10)

    return train_data, memory_data, test_data


def get_shadow_stl10(args):
    training_data_num = 50000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)

    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir + "train_unlabeled.npz",
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,indices = training_data_sampling_indices,
        transform=train_transform,
        bd_transform=backdoor_transform,
        ftt_transform=finetune_transform
    )
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.pretraining_dataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.pretraining_dataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.pretraining_dataset == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.pretraining_dataset == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor


def get_downstream_stl10(args):
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
