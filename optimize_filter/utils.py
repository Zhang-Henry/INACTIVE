import sys
sys.path.append("..")
from torch.autograd import Function
import torch.nn as nn
import torch
import numpy as np
from torch.nn import Identity
from optimize_filter.PyTorch_CIFAR10.cifar10_models.resnet import resnet34

import gc

"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)


        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

def cosine_similarity(P, Q):
    norm_P = torch.norm(P)
    norm_Q = torch.norm(Q)

    # Normalize P and Q
    normalized_P = P / norm_P
    normalized_Q = Q / norm_Q

    # Compute the cosine similarity
    similarity = torch.dot(normalized_P, normalized_Q)

    return similarity.item()  # Convert the result to a Python scalar

def kl_divergence(P, Q):
    # Normalize P and Q
    P_normalized = P / P.sum()
    Q_normalized = Q / Q.sum()

    # Calculate the element-wise KL divergence
    kl_elements = P_normalized * torch.log(P_normalized / Q_normalized)

    # Sum up the element-wise KL divergence
    kl_div = torch.sum(kl_elements)

    return kl_div.item()  # Convert the result to a Python scalar



# https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 4.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = 1.0
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    """
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).cuda()  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-2)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class Recorder:
    def __init__(self, args):
        super().__init__()

        # Best optimization results
        # self.mask_best = None
        # self.pattern_best = None
        self.best = -float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.best

        # Cost
        self.cost = args.init_cost
        self.cost_multiplier_up = args.cost_multiplier_up
        self.cost_multiplier_down = args.cost_multiplier_down
        self.max_cost = args.max_cost
        self.min_cost = args.min_cost

    def reset_state(self, args):
        self.cost = args.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))


class Loss:
    def __init__(self):
        super().__init__()
        self.losses=[]
        self.avg=0.0

    def update(self,num):
        self.losses.append(num)
        self.avg=np.mean(self.losses)

    def reset(self):
        self.losses=[]
        self.avg=0.0


# class Loss_Tracker:
#     def __init__(self) -> None:
#         self.loss=Loss()
#         self.wd=Loss()
#         self.ssim=Loss()
#         self.psnr=Loss()
#         self.lp=Loss()
#         self.sim=Loss()
#         self.far=Loss()

#     def update(self,loss,wd,ssim,psnr,lp,sim,far,):
#         self.loss.update(loss)
#         self.wd.update(wd)
#         self.ssim.update(ssim)
#         self.psnr.update(psnr)
#         self.lp.update(lp)
#         self.sim.update(sim)
#         self.far.update(far)

#     def get_avg_loss(self):
#         return self.loss.avg,self.wd.avg,self.ssim.avg,self.psnr.avg,self.lp.avg,self.sim.avg,self.far.avg

#     def reset(self):
#         self.loss.reset()
#         self.wd.reset()
#         self.ssim.reset()
#         self.psnr.reset()
#         self.lp.reset()
#         self.sim.reset()
#         self.far.reset()

class Loss_Tracker:
    def __init__(self, loss_names):
        self.losses = {name: Loss() for name in loss_names}

    def update(self, losses):
        for name, value in losses.items():
            self.losses[name].update(value)

    def get_avg_loss(self):
        return {name: loss.avg for name, loss in self.losses.items()}

    def reset(self):
        for loss in self.losses.values():
            loss.reset()

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, resnet):
        super(ResNetFeatureExtractor, self).__init__()
        # 将ResNet的层分开
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        # 提取各个层的特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_out = x

        x = self.maxpool(x)
        x = self.layer1(x)
        layer1_out = x

        x = self.layer2(x)
        layer2_out = x

        x = self.layer3(x)
        layer3_out = x

        x = self.layer4(x)
        layer4_out = x

        # 返回提取的特征
        return conv1_out, layer1_out, layer2_out, layer3_out, layer4_out
        # return layer4_out




def load_backbone():
    backbone = resnet34(pretrained=True)
    backbone.fc = Identity()

    # model = resnet34(pretrained=True)
    # backbone = ResNetFeatureExtractor(model)
    # del model
    backbone.eval() # for evaluation
    return backbone