import sys
sys.path.append('..')

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import ToTensor

import copy,os

class ReferenceImg(Dataset):

    def __init__(self, reference_file, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_input_array = np.load(reference_file)

        self.data = self.target_input_array['x']
        self.targets = self.target_input_array['y']

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class BadEncoderDataset(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']

        # self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file)

        # self.trigger_patch_list = self.trigger_input_array['t']
        # self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x']

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

        # self.state_dict = torch.load(trigger_file, map_location=torch.device('cpu'))
        # self.net = U_Net_tiny(img_ch=3,output_ch=3)
        # self.net.load_state_dict(self.state_dict['model_state_dict'])
        # print(summary(self.net, (3, 32, 32), device='cpu'))
        # self.net=self.net.eval()


    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''

        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            img_backdoor = self.bd_transform(img_copy)
            img_backdoor_list.append(img_backdoor)


        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return,im_1

    def __len__(self):
        return len(self.indices)



class BadEncoderTestBackdoor(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_label, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']


        # self.trigger_input_array = np.load(trigger_file)

        # self.trigger_patch_list = self.trigger_input_array['t']
        # self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label

        self.test_transform = transform

        # state_dict = torch.load(trigger_file, map_location=torch.device('cpu'))
        # self.net = U_Net_tiny(img_ch=3,output_ch=3)
        # self.net.load_state_dict(state_dict['model_state_dict'])
        # self.net=self.net.eval()

        # self.filter = torch.load('trigger/filter.pt', map_location=torch.device('cpu'))

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])

        ###########################
        ### for ins filter only ###

        # image_pil = Image.fromarray(img)
        # filtered_image_pil = pilgram.xpro2(image_pil)
        # img_backdoor =self.test_transform(filtered_image_pil)

        ###########################

        # img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        # img_backdoor =self.test_transform(Image.fromarray(img))


        ###########################
        # for ctrl only
        # trans=transforms.Compose([
        #         transforms.ToTensor(),
        #     ])

        # image_pil = Image.fromarray(img)
        # tensor_image = trans(image_pil)

        # base_image=tensor_image.unsqueeze(0)
        # poison_frequency_agent = PoisonFre('args',32, [1,2], 32, [15,31],  False,  True)

        # x_tensor,_ = poison_frequency_agent.Poison_Frequency_Diff(base_image,0, 100.0)
        # img_backdoor = x_tensor.squeeze()

        # # img_backdoor = np.clip(img_backdoor, 0, 1) #限制颜色范围在0-1

        # img_backdoor = self.test_transform(img_backdoor.permute(1,2,0).detach().numpy())


        ########################
        img = Image.fromarray(img)
        img_backdoor =self.test_transform(img)

        return img_backdoor, self.target_class


    def __len__(self):
        return self.data.shape[0]



class CIFAR10CUSTOM(Dataset):

    def __init__(self, numpy_file, class_type, transform=None, transform2=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()
        self.classes = class_type
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return self.data.shape[0]


class CIFAR10Pair(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class CIFAR10Mem(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CIFAR10M(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img_trans = self.transform(img)

        return self.transform2(img), img_trans


class BadEncoderTestBackdoor_224(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_label, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = []
        self.targets = []
        for filename in os.listdir(numpy_file):
            if filename.endswith('.jpeg'):
                self.data.append(os.path.join(numpy_file, filename))
                # Extract label from filename
                label = int(filename.split('_')[3].replace('.jpeg', '')[1:-1])
                self.targets.append(label)

        self.target_class = reference_label
        self.test_transform = transform

        # self.trigger_input_array = np.load(trigger_file)

        # self.trigger_patch_list = self.trigger_input_array['t']
        # self.trigger_mask_list = self.trigger_input_array['tm']

        # state_dict = torch.load(trigger_file, map_location=torch.device('cpu'))
        # self.net = U_Net_tiny(img_ch=3,output_ch=3)
        # self.net.load_state_dict(state_dict['model_state_dict'])

    def __getitem__(self,index):
        img = Image.open(self.data[index])

        ###########################
        ### for ins filter only ###

        # image_pil = Image.fromarray(img)
        # filtered_image_pil = pilgram.kelvin(image_pil)
        # img_backdoor =self.test_transform(filtered_image_pil)

        ###########################
        # img = np.array(img)
        # img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        # img_backdoor =self.test_transform(Image.fromarray(img))

        ###########################
        # for customized filter only

        # img_copy=torch.Tensor(img)
        # backdoored_image = F.conv2d(img_copy.permute(2, 0, 1), self.filter, padding=7//2)
        # img_backdoor = self.test_transform(backdoored_image.permute(1,2,0).detach().numpy())

        ###########################
        ###########################
        # for ctrl only
        # trans=transforms.Compose([
        #         transforms.ToTensor()
        #     ])

        # image_pil = Image.fromarray(img)
        # tensor_image = trans(image_pil)

        # base_image=tensor_image.unsqueeze(0)
        # poison_frequency_agent = PoisonFre('args',32, [1,2], 32, [15,31],  False,  True)

        # x_tensor,_ = poison_frequency_agent.Poison_Frequency_Diff(base_image,0, 100.0)
        # img_backdoor = x_tensor.squeeze()

        # img_backdoor = np.clip(img_backdoor, 0, 1) #限制颜色范围在0-1

        # img_backdoor = self.bd_transform(img_backdoor.detach().numpy())

        ###########################
        # unet
        # img = self.test_transform(img)
        # img_backdoor_=self.net(img.unsqueeze(0))
        # img_backdoor = img_backdoor_.squeeze()
        # img_backdoor = torch.clamp(img_backdoor, min=0, max=1)

        ################################
        # to_tensor = ToTensor()
        # tensor_image = to_tensor(img)
        # img=np.array(img)
        # tensor_image = torch.Tensor(img)
        # backdoored_image=self.net(tensor_image.permute(2, 0, 1).unsqueeze(0))
        # img_backdoor = backdoored_image.squeeze()
        # img_backdoor = self.test_transform(img_backdoor.permute(1,2,0).detach().numpy())
        ################################

        img_backdoor = self.test_transform(img)
        ###########################
        return img_backdoor, self.target_class


    def __len__(self):
        return len(self.data)



class CIFAR10CUSTOM_224(Dataset):

    def __init__(self, numpy_file, class_type, transform=None, transform2=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = []
        self.targets = []
        for filename in os.listdir(numpy_file):
            if filename.endswith('.jpeg'):
                self.data.append(os.path.join(numpy_file, filename))
                # Extract label from filename
                label = int(filename.split('_')[3].replace('.jpeg', '')[1:-1])
                self.targets.append(label)

        self.classes = class_type
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)
        # return self.data.shape[0]


class CIFAR10Pair_224(CIFAR10CUSTOM_224):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        # img = Image.fromarray(img)
        img = Image.open(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class CIFAR10Mem_224(CIFAR10CUSTOM_224):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CIFAR10M_224(CIFAR10CUSTOM_224):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)
        img = Image.open(img)

        if self.transform is not None:
            img_trans = self.transform(img)

        return self.transform2(img), img_trans


### 读取gtrsb_224,cifar10_224，stl10_224数据集
class CustomDataset_224(Dataset):
    def __init__(self, directory, transform1=None, transform2=None):
        self.clean_transform = transform1
        self.bd_transform = transform2
        self.images = []
        self.labels = []
        for filename in os.listdir(directory):
            if filename.endswith('.jpeg'):
                self.images.append(os.path.join(directory, filename))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        img_raw = self.clean_transform(image)
        img_trans = self.bd_transform(image)

        return img_raw,img_trans


class CustomDataset_label(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        for filename in os.listdir(directory):
            if filename.endswith('.jpeg'):
                self.images.append(os.path.join(directory, filename))
                # Extract label from filename
                label = int(filename.split('_')[3].replace('.jpeg', '')[1:-1])
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

