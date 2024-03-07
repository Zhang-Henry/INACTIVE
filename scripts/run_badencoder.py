import os
from datetime import datetime

# 获取当前时间
now = datetime.now()
time =  now.strftime("%Y-%m-%d-%H:%M:%S")

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')


def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, pretraining_dataset, bz,color,loss0, clean_encoder='model_1000.pth'):
    save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder/{time}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    # os.makedirs(f'{save_path}/{time}')
    # filter_path="optimize_filter/trigger/unet_filter.pt"

    cmd = f'nohup python3 -u badencoder.py \
    --epochs 200 \
    --timestamp {time} \
    --lr 0.001 \
    --batch_size {bz}   \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder ./output/{encoder_usage_info}/clean_encoder/{clean_encoder} \
    --encoder_usage_info {encoder_usage_info} \
    --gpu {gpu} \
    --reference_file ./reference/{encoder_usage_info}/{reference}.npz \
    --trigger_file {trigger} \
    --pretraining_dataset {pretraining_dataset} \
    --color {color} \
    --loss0 {loss0} \
    > ./log/bad_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}.log 2>&1 &'
    os.system(cmd)


run_finetune(0, 'cifar10', 'cifar10', 'stl10', 'optimize_filter/trigger/cifar10/2023-12-06-23-41-20/ssim0.9328_psnr22.50_lp0.0291_wd0.603_color11.353.pt', 'truck','cifar10',512,0.1,10)
# run_finetune(1, 'cifar10', 'cifar10', 'gtsrb', 'optimize_filter/trigger/cifar10/2023-12-06-23-41-20/ssim0.9328_psnr22.50_lp0.0291_wd0.603_color11.353.pt', 'priority','cifar10',256,0,10)
# run_finetune(5, 'cifar10', 'cifar10', 'svhn', 'optimize_filter/trigger/cifar10/2023-12-06-23-41-20/ssim0.9328_psnr22.50_lp0.0291_wd0.603_color11.353.pt', 'one','cifar10',64,0.1,10)




