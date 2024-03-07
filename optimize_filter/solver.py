import torch,lpips
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
from pytorch_ssim import SSIM
from tqdm import tqdm
from PIL import Image
from utils import *
from loss import *
from network import U_Net,R2AttU_Net,R2U_Net,AttU_Net
from tiny_network import U_Net_tiny
# from optimize_filter.previous.data_loader import aug
from torchmetrics.image import PeakSignalNoiseRatio


class Solver():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = U_Net_tiny(img_ch=3,output_ch=3).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        # self.filter = torch.nn.Parameter(torch.randn(3, 3, 7, 7, requires_grad=True).cuda())  # 修改滤波器形状并将其放在GPU上
        # self.optimizer = torch.optim.Adam([self.filter], lr=args.lr)

        self.scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

        self.ssim = SSIM()
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.mse = nn.MSELoss()
        self.WD=SinkhornDistance(eps=0.1, max_iter=100)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.color_loss_fn = CombinedColorLoss().to(self.device)
        self.backbone = load_backbone()
        self.backbone = self.backbone.to(self.device).eval()
        # print(self.backbone)
        # print(self.net)


    def train(self,args,train_loader,test_loader=None):
        print('Start training...')

        bar=tqdm(range(1, args.n_epoch+1))
        recorder=Recorder(args)
        tracker=Loss_Tracker(['loss', 'wd', 'ssim', 'psnr','mse', 'lp', 'sim', 'far','color'])
        tracker_test=Loss_Tracker(['wd', 'ssim', 'psnr','mse', 'lp','color'])
#

        # 恢复模型和优化器状态
        if args.resume:
            checkpoint = torch.load(args.resume)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            recorder.best = checkpoint['best']
            print(f"\nResuming training from {args.resume}")

        for _ in bar:
            self.train_one_epoch(args,recorder,bar,tracker,train_loader)
            if args.dataset != 'imagenet':
                self.test_one_epoch(args,test_loader,tracker_test)
            # self.test_one_epoch(args,test_loader,tracker_test)


    def train_one_epoch(self,args,recorder,bar,tracker,train_loader):
        tracker.reset() # 重置损失记录器
        self.net.train()

        for img,img_trans in train_loader:
            img = img.to(self.device)
            img_trans = img_trans.to(self.device)

            # 将滤镜作用在Aug的图像上
            # filter_img = self.net(img_trans)
            filter_img = self.net(img)

            if args.dataset=='cifar10':
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).cuda()
                std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).cuda()

                filter_img = filter_img * std + mean # denormalize
                img = img * std + mean
                img_trans = img_trans * std + mean

            elif args.dataset=='stl10':
                mean = torch.tensor([0.44087798, 0.42790666, 0.38678814]).view(1, 3, 1, 1).cuda()
                std = torch.tensor([0.25507198, 0.24801506, 0.25641308]).view(1, 3, 1, 1).cuda()
                filter_img = filter_img * std + mean # denormalize
                img = img * std + mean
                img_trans = img_trans * std + mean
            elif args.dataset=='imagenet':
                # mean = torch.tensor([0.4850, 0.4560, 0.4060]).view(1, 3, 1, 1).cuda()
                # std = torch.tensor([0.2290, 0.2240, 0.2250]).view(1, 3, 1, 1).cuda()
                pass


            # sig=torch.nn.Sigmoid()
            # filter_img = sig(filter_img)

            filter_img = torch.clamp(filter_img, min=0, max=1)

            color_loss = self.color_loss_fn(filter_img, img_trans, args)

            with torch.no_grad():
                img_feature = self.backbone(img)
                filter_img_feature = self.backbone(filter_img)

                img_feature = F.normalize(img_feature, dim=-1)
                filter_img_feature = F.normalize(filter_img_feature, dim=-1)
                wd,_,_=self.WD(filter_img_feature,img_feature) # wd越小越相似，拉远backdoor img和transformed backdoor img的距离
                # wd = compute_style_loss(filter_img_feature,img_trans_feature)

            # filter后的图片和原图的mse和ssim，差距要尽可能小


            loss_psnr = self.psnr(filter_img, img)
            loss_ssim = self.ssim(filter_img, img)
            # loss_mse = self.mse(filter_img, img)


            d_list = self.loss_fn(filter_img,img)
            lp_loss=d_list.squeeze()


            if args.ablation:
                loss_sim = 10 * lp_loss.mean() + wd
                loss_far = recorder.cost * (1 - loss_ssim - 0.025 * loss_psnr)
                loss = loss_sim + loss_far
            elif args.most_close:
                loss_sim = wd + 1 - loss_ssim  + 10 * lp_loss.mean() - 0.025 * loss_psnr
                loss_far = 0 * loss_sim
                loss=loss_sim
            else:
                loss_sim = 1 - loss_ssim + 10 * lp_loss.mean() - 0.025 * loss_psnr + wd
                loss_far = - recorder.cost * color_loss
                loss = loss_sim + loss_far

            self.optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)

            self.optimizer.step()
            # losses={'loss':loss.item(),'wd':wd.item(),'ssim':loss_ssim.item(),'psnr':loss_psnr.item(),'mse':loss_mse.item(),'lp':lp_loss.mean().item(),'sim':loss_sim.item(),'far':loss_far.item(),'color':color_loss.item()}
            losses={'loss':loss.item(),'wd':wd.item(),'ssim':loss_ssim.item(),'psnr':loss_psnr.item(),'lp':lp_loss.mean().item(),'sim':loss_sim.item(),'far':loss_far.item(),'color':color_loss.item()}

            tracker.update(losses)

        self.scheduler.step()
        # 计算平均损失

        avg_losses = tracker.get_avg_loss()
        avg_loss,wd, ssim, psnr, mse, lp, sim, far, color = avg_losses.values()

        if args.ablation or args.most_close:
            if ssim >= args.ssim_threshold:
                state = {
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best': recorder.best
                }
                torch.save(state, f'trigger/{args.dataset}/{self.args.timestamp}/ablation_ssim{ssim:.4f}_psnr{psnr:.2f}_lp{lp:.4f}_wd{wd:.3f}_color{color:.3f}.pt')

                recorder.best = ssim
                print('\n--------------------------------------------------')
                print(f"Updated !!! Best sim:{sim}, far:{far}, SSIM: {ssim}, psnr: {psnr}, lp: {lp}, WD: {wd}, color: {color}")
                print('--------------------------------------------------')
                recorder.cost_up_counter = 0
                recorder.cost_down_counter = 0

            if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold:
                recorder.cost_up_counter += 1
                recorder.cost_down_counter = 0
            else:
                recorder.cost_up_counter = 0
                recorder.cost_down_counter += 1

            if recorder.cost_up_counter >= args.patience:
                recorder.cost_up_counter = 0
                print('\n--------------------------------------------------')
                print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
                print('--------------------------------------------------')

                recorder.cost *= recorder.cost_multiplier_up
                recorder.cost_up_flag = True

            elif recorder.cost_down_counter >= args.patience:
                recorder.cost_down_counter = 0
                print('\n--------------------------------------------------')
                print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
                print('--------------------------------------------------')
                recorder.cost /= recorder.cost_multiplier_down
                recorder.cost_down_flag = True
        else: # 正常情况
            if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold and color >= recorder.best:
                state = {
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best': recorder.best
                }
                torch.save(state, f'trigger/{args.dataset}/{self.args.timestamp}/ssim{ssim:.4f}_psnr{psnr:.2f}_lp{lp:.4f}_wd{wd:.3f}_color{color:.3f}.pt')

                recorder.best = color
                print('\n--------------------------------------------------')
                print(f"Updated !!! Best sim:{sim}, far:{far}, SSIM: {ssim}, psnr: {psnr}, lp: {lp}, WD: {wd}, color: {color}")
                print('--------------------------------------------------')
                recorder.cost_up_counter = 0
                recorder.cost_down_counter = 0


            if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold:
                recorder.cost_up_counter += 1
                recorder.cost_down_counter = 0
            else:
                recorder.cost_up_counter = 0
                recorder.cost_down_counter += 1

            if recorder.cost_up_counter >= args.patience:
                recorder.cost_up_counter = 0
                print('\n--------------------------------------------------')
                print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
                print('--------------------------------------------------')

                recorder.cost *= recorder.cost_multiplier_up
                recorder.cost_up_flag = True

            elif recorder.cost_down_counter >= args.patience:
                recorder.cost_down_counter = 0
                print('\n--------------------------------------------------')
                print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
                print('--------------------------------------------------')
                recorder.cost /= recorder.cost_multiplier_down
                recorder.cost_down_flag = True

        bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim:.5f}, far:{far:.5f}, WD: {wd:.5f}, SSIM: {ssim:.5f}, pnsr:{psnr:.5f}, mse:{mse:5f}, lp:{lp:.5f}, color:{color:.5f},  cost:{recorder.cost}")


    def test_one_epoch(self,args,test_loader,tracker_test):
        tracker_test.reset()
        self.net.eval()
        with torch.no_grad():
            for img,img_trans in test_loader:
                img = img.to(self.device)
                img_trans = img_trans.to(self.device)

                filter_img = self.net(img_trans)
                filter_img = self.net(img)

                if args.dataset=='cifar10':
                    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).cuda()
                    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).cuda()

                elif args.dataset=='stl10':
                    mean = torch.tensor([0.44087798, 0.42790666, 0.38678814]).view(1, 3, 1, 1).cuda()
                    std = torch.tensor([0.25507198, 0.24801506, 0.25641308]).view(1, 3, 1, 1).cuda()

                elif args.dataset=='imagenet' or args.dataset=='imagenet_gtsrb_stl10_svhn':
                    mean = torch.tensor([0.4850, 0.4560, 0.4060]).view(1, 3, 1, 1).cuda()
                    std = torch.tensor([0.2290, 0.2240, 0.2250]).view(1, 3, 1, 1).cuda()

                filter_img = filter_img * std + mean # denormalize
                img = img * std + mean
                img_trans = img_trans * std + mean

                # sig=torch.nn.Sigmoid()
                # filter_img = sig(filter_img)

                color_loss = self.color_loss_fn(filter_img, img_trans,args)
                img_trans_feature = self.backbone(img_trans)
                filter_img_feature = self.backbone(filter_img)

                img_trans_feature = F.normalize(img_trans_feature, dim=-1)
                filter_img_feature = F.normalize(filter_img_feature, dim=-1)
                wd,_,_=self.WD(filter_img_feature,img_trans_feature) # wd越小越相似，拉远backdoor img和transformed backdoor img的距离

                loss_psnr = self.psnr(filter_img, img)
                loss_ssim = self.ssim(filter_img, img)
                # loss_mse = self.mse(filter_img, img)

                d_list = self.loss_fn(filter_img,img)
                lp_loss=d_list.squeeze()

                # losses={'wd':wd.item(),'ssim':loss_ssim.item(),'psnr':loss_psnr.item(),'mse':loss_mse.item(),'lp':lp_loss.mean().item(),'color':color_loss.item()}
                losses={'wd':wd.item(),'ssim':loss_ssim.item(),'psnr':loss_psnr.item(),'lp':lp_loss.mean().item(),'color':color_loss.item()}

                tracker_test.update(losses)

            avg_losses = tracker_test.get_avg_loss()
            wd, ssim, psnr, mse, lp, color = avg_losses.values()
            print(f"\nTEST: WD: {wd:.5f}, SSIM: {ssim:.5f}, pnsr:{psnr:.5f}, lp:{lp:.5f}, color:{color:.5f}")

