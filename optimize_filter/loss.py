import torch
# from .utils import *
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from kornia.color import rgb_to_hsv,rgb_to_hls
import torch.nn as nn





class ColorLoss(torch.nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, generated_img, original_img):
        # 转换颜色空间
        original_hsv = rgb_to_hsv(original_img)
        generated_hsv = rgb_to_hsv(generated_img)

        # 计算HSV通道的差异
        hue_loss = F.mse_loss(original_hsv[:, :, 0], generated_hsv[:, :, 0])
        saturation_loss = F.mse_loss(original_hsv[:, :, 1], generated_hsv[:, :, 1])
        value_loss = F.mse_loss(original_hsv[:, :, 2], generated_hsv[:, :, 2])

        # 综合三个通道的损失
        total_loss = hue_loss + saturation_loss + value_loss
        return total_loss


# class ColorLoss(torch.nn.Module):
#     def __init__(self):
#         super(ColorLoss, self).__init__()

#     def forward(self, original_img, generated_img):
#         # 转换颜色空间
#         original_hls = rgb_to_hls(original_img)
#         generated_hls = rgb_to_hls(generated_img)

#         # 计算HLS通道的差异
#         hue_loss = torch.nn.functional.l1_loss(original_hls[:, :, 0], generated_hls[:, :, 0])
#         lightness_loss = torch.nn.functional.l1_loss(original_hls[:, :, 1], generated_hls[:, :, 1])
#         saturation_loss = torch.nn.functional.l1_loss(original_hls[:, :, 2], generated_hls[:, :, 2])

#         # 综合三个通道的损失
#         total_loss = hue_loss + lightness_loss + saturation_loss
#         return total_loss


# class CombinedColorLoss(torch.nn.Module):
    # def __init__(self):
    #     super(CombinedColorLoss, self).__init__()

    # def forward(self, original_img, generated_img):
    #     # HLS 转换和损失计算
    #     original_hls = rgb_to_hls(original_img)
    #     generated_hls = rgb_to_hls(generated_img)
    #     hls_loss = F.mse_loss(original_hls, generated_hls)

    #     # HSV 转换和损失计算
    #     original_hsv = rgb_to_hsv(original_img)
    #     generated_hsv = rgb_to_hsv(generated_img)
    #     hsv_loss = F.mse_loss(original_hsv, generated_hsv)

    #     # 组合 HLS 和 HSV 损失
    #     total_loss = hls_loss + hsv_loss
    #     return total_loss

# class CharbonnierLoss(nn.Module):
#     def __init__(self, epsilon=1e-3):
#         super(CharbonnierLoss, self).__init__()
#         self.epsilon = epsilon

#     def forward(self, predicted, target):
#         return torch.mean(torch.sqrt((predicted - target) ** 2 + self.epsilon ** 2))


class CombinedColorLoss(torch.nn.Module):
    def __init__(self):
        super(CombinedColorLoss, self).__init__()
        # self.huber_loss = nn.SmoothL1Loss()
        # self.charbonnier_loss = CharbonnierLoss()
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    def forward(self, original_img, generated_img, args):
        # 转换到HSV和HLS颜色空间
        original_hsv = rgb_to_hsv(original_img)
        generated_hsv = rgb_to_hsv(generated_img)
        original_hls = rgb_to_hls(original_img)
        generated_hls = rgb_to_hls(generated_img)

        # 计算HSV和HLS中的Hue和Saturation损失
        hue_loss_hsv = F.mse_loss(original_hsv[:, 0, :, :], generated_hsv[:, 0, :, :])
        saturation_loss_hsv = F.mse_loss(original_hsv[:, 1, :, :], generated_hsv[:, 1, :, :])
        # hue_loss_hls = F.mse_loss(original_hls[:, 0, :, :], generated_hls[:, 0, :, :])

        lightness_loss = F.mse_loss(original_hls[:, 1, :, :], generated_hls[:, 1, :, :])

        # 计算HSV的明度(V)损失和HLS的亮度(L)损失
        value_loss_hsv = F.mse_loss(original_hsv[:, 2, :, :], generated_hsv[:, 2, :, :])

        # saturation_loss_hls = F.mse_loss(original_hls[:, 2, :, :], generated_hls[:, 2, :, :])
        # 综合损失
        total_loss = args.hue_hsv * hue_loss_hsv + args.saturation_hsv * saturation_loss_hsv + \
        args.value_hsv * value_loss_hsv + args.lightness * lightness_loss

        return total_loss
