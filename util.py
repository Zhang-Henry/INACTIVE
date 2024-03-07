
import torch,lpips
import torch.nn.functional as F


from optimize_filter.network import AttU_Net
from optimize_filter.tiny_network import U_Net_tiny

from optimize_filter.utils import SinkhornDistance, Recorder, Loss_Tracker
from torch.nn import MSELoss
from pytorch_ssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from datetime import datetime
from loss import *
from optimize_filter.utils import load_backbone

WD=SinkhornDistance(eps=0.1, max_iter=100)
ssim = SSIM()
loss_fn = lpips.LPIPS(net='alex').cuda()
psnr = PeakSignalNoiseRatio().cuda()
color_loss_fn = CombinedColorLoss().cuda()
backbone = load_backbone()
backbone = backbone.cuda().eval()

def filter_color_loss(filter,img_clean,img_trans,tracker,loss_0,args):
    # metric definition
    filter.train()

    filter_img = filter(img_clean) # backdoor img

    if args.shadow_dataset=='cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).cuda()

    elif args.shadow_dataset=='stl10':
        mean = torch.tensor([0.44087798, 0.42790666, 0.38678814]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.25507198, 0.24801506, 0.25641308]).view(1, 3, 1, 1).cuda()

    elif args.shadow_dataset=='imagenet' or args.shadow_dataset=='imagenet_gtsrb_stl10_svhn':
        mean = torch.tensor([0.4850, 0.4560, 0.4060]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.2290, 0.2240, 0.2250]).view(1, 3, 1, 1).cuda()

    filter_img = filter_img * std + mean # denormalize
    img_clean = img_clean * std + mean
    img_trans = img_trans * std + mean

    filter_img = torch.clamp(filter_img, min=0, max=1)

    img_clean_feature = backbone(img_clean)
    filter_img_feature = backbone(filter_img)

    img_clean_feature = F.normalize(img_clean_feature, dim=-1)
    filter_img_feature = F.normalize(filter_img_feature, dim=-1)
    wd,_,_=WD(filter_img_feature,img_clean_feature) # wd越小越相似，拉远backdoor img和transformed backdoor img的距离


    loss_psnr = psnr(filter_img, img_clean)
    loss_ssim = ssim(filter_img, img_clean)
    d_list = loss_fn(filter_img,img_clean)
    lp_loss=d_list.squeeze()

    color_loss = color_loss_fn(filter_img, img_trans, args)
    loss_sim = 1 - loss_ssim - args.psnr * loss_psnr + args.loss0 * loss_0 + wd + 10 * lp_loss.mean()

    loss_far = args.color * color_loss

    # loss = - loss_far
    loss = loss_sim - loss_far

    # print(f'\nloss:{loss},loss_sim:{loss_sim}, loss_far:{loss_far}, wd:{wd},ssim:{loss_ssim},lp:{lp_loss.mean()},psnr:{loss_psnr},color:{color_loss},cost:{recorder.cost}')
    losses={'loss':loss.item(),'wd':wd.item(),'ssim':loss_ssim.item(),'psnr':loss_psnr.item(),'lp':lp_loss.mean().item(),'sim':loss_sim.item(),'far':loss_far.item(),'color':color_loss.item()}
    print('')
    print(losses)
    tracker.update(losses)
    return loss



def clamp_batch_images(batch_images, args):
    """
    Clamps each channel of a batch of images within the range defined by the mean and std.

    Parameters:
    batch_images (Tensor): A batch of images, shape [batch_size, channels, height, width].
    mean (list): A list of mean for each channel.
    std (list): A list of standard deviations for each channel.

    Returns:
    Tensor: The batch of clamped images.
    """
    # 获取通道数
    shadow_dataset = getattr(args, 'shadow_dataset', None)
    dataset = getattr(args, 'encoder_usage_info', None)

    if shadow_dataset:
        dataset_name = shadow_dataset
    elif dataset:
        dataset_name = dataset
    else:
        dataset_name = None

    if dataset_name=='cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
        std = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
    elif dataset_name=='stl10':
        mean = torch.tensor([0.44087798, 0.42790666, 0.38678814]).cuda()
        std = torch.tensor([0.25507198, 0.24801506, 0.25641308]).cuda()
    elif dataset_name=='imagenet':
        mean = torch.tensor([0.4850, 0.4560, 0.4060]).cuda()
        std = torch.tensor([0.2290, 0.2240, 0.2250]).cuda()

    # 确保均值和标准差列表长度与通道数匹配
    num_channels =batch_images.shape[1]
    if len(mean) != num_channels or len(std) != num_channels:
        raise ValueError("The length of mean and std must match the number of channels")

    # 创建一个相同形状的张量用于存放裁剪后的图像

    clamped_images = torch.empty_like(batch_images)

    # 对每个通道分别进行裁剪
    for channel in range(batch_images.shape[1]):
        min_val = (0 - mean[channel]) / std[channel]
        max_val = (1 - mean[channel]) / std[channel]
        clamped_images[:, channel, :, :] = torch.clamp(batch_images[:, channel, :, :], min=min_val, max=max_val)

    return clamped_images