from torchvision import transforms
from .backdoor_dataset import *
import numpy as np
from .trans import *



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




def get_downstream_gtsrb(args):
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
        memory_data = CIFAR10Mem_224(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        test_data_backdoor = BadEncoderTestBackdoor_224(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        test_data_clean = CIFAR10Mem_224(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        # training_file_name = 'train_224.npz'
        # testing_file_name = 'test_224.npz'
        # memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        # test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        # test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
        training_file_name = 'train_224'
        testing_file_name = 'test_224'
        memory_data = CIFAR10Mem_224(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        test_data_backdoor = BadEncoderTestBackdoor_224(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        test_data_clean = CIFAR10Mem_224(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)


    return target_dataset, memory_data, test_data_clean, test_data_backdoor


