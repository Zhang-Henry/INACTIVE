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

    cmd = f'nohup python3 -u imperative.py \
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




run_finetune(5, 'stl10', 'stl10', 'cifar10', 'trigger/stl10/ssim0.9182_psnr22.37_lp0.0263_wd0.702_color10.051.pt', 'airplane', 'stl10',256,0.1,5)
# run_finetune(3, 'stl10', 'stl10', 'gtsrb', 'trigger/stl10/ssim0.9182_psnr22.37_lp0.0263_wd0.702_color10.051.pt', 'priority', 'stl10',256,0.1,5)
# run_finetune(2, 'stl10', 'stl10', 'svhn', 'trigger/stl10/ssim0.9182_psnr22.37_lp0.0263_wd0.702_color10.051.pt', 'one', 'stl10',256,0.1,5)
