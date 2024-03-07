timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

# nohup python main.py \
#     --timestamp $timestamp \
#     --gpu 1 \
#     --batch_size 38 \
#     --ssim_threshold 0.75 \
#     --n_epoch 300 \
#     --step_size 50 \
#     --patience 5 \
#     --init_cost 0.0001 \
#     --cost_multiplier_up 1.25 \
#     --cost_multiplier_down 1.2 \
#     > logs/moco/filter_AttU_Net_wd_lpips_$timestamp.log 2>&1 &

# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.005 \
#     --gpu 2 \
#     --batch_size 38 \
#     --ssim_threshold 0.75 \
#     --psnr_threshold 15.0 \
#     --lp_threshold 0.5 \
#     --n_epoch 150 \
#     --step_size 50 \
#     --patience 5 \
#     --init_cost 0.00025 \
#     --cost_multiplier_up 1.04 \
#     --cost_multiplier_down 1.08 \
#     --resume /home/hrzhang/projects/SSL-Backdoor/optimize_filter/trigger/moco/2023-11-01-19-07-19/ssim0.8186_psnr22.45_lp0.0845_wd23291.713.pt \
#     > logs/moco/filter_nofeature_$timestamp.log 2>&1 &

######### use feature #########
# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.005 \
#     --gpu 3 \
#     --batch_size 1024 \
#     --ssim_threshold 0.95 \
#     --psnr_threshold 30.0 \
#     --lp_threshold 0.01 \
#     --n_epoch 150 \
#     --step_size 50 \
#     --patience 5 \
#     --init_cost 3 \
#     --cost_multiplier_up 1.5 \
#     --cost_multiplier_down 2 \
#     --use_feature \
#     > logs/cifar10/filter_$timestamp.log 2>&1 &




# nohup python main.py \
#     --ablation True \
#     --timestamp $timestamp \
#     --gpu 2 \
#     --batch_size 32 \
#     --init_cost 1.2 \
#     --cost_multiplier_up 2 \
#     --cost_multiplier_down 2 \
#     --patience 3 \
#     --ssim_threshold 0.90 \
#     --n_epoch 50 \
#     --step_size 30 \
#     > logs/moco/filter_unet_wd_ablation_$timestamp.log 2>&1 &


######### NOT use feature #########
# 跟原图越接近，wd越近
# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.005 \
#     --gpu 3 \
#     --batch_size 1024 \
#     --ssim_threshold 0.95 \
#     --psnr_threshold 35.0 \
#     --lp_threshold 0.01 \
#     --n_epoch 150 \
#     --step_size 50 \
#     --patience 5 \
#     --init_cost 0.0001 \
#     --cost_multiplier_up 2 \
#     --cost_multiplier_down 3 \
#     > logs/cifar10/filter_nofeature_$timestamp.log 2>&1 &

# one layer loss stl10
# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.01 \
#     --gpu 1 \
#     --batch_size 512 \
#     --ssim_threshold 0.90 \
#     --psnr_threshold 22.0 \
#     --lp_threshold 0.02 \
#     --n_epoch 400 \
#     --step_size 150 \
#     --patience 5 \
#     --init_cost 0.01 \
#     --cost_multiplier_up 3 \
#     --cost_multiplier_down 1.5 \
#     --dataset 'stl10' \
#     > logs/stl10/one_layer_filter_color_wd_$timestamp.log 2>&1 &

##### color loss #####

# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.01 \
#     --gpu 5 \
#     --batch_size 2048 \
#     --ssim_threshold 0.80 \
#     --psnr_threshold 13.0 \
#     --lp_threshold 0.1 \
#     --n_epoch 200 \
#     --step_size 100 \
#     --patience 5 \
#     --init_cost 1 \
#     --cost_multiplier_up 1.5 \
#     --cost_multiplier_down 2 \
#     --dataset 'cifar10' \
#     > logs/cifar10/filter_color_wd_$timestamp.log 2>&1 &



# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.01 \
#     --gpu 5 \
#     --batch_size 2048 \
#     --ssim_threshold 0.90 \
#     --psnr_threshold 20.0 \
#     --lp_threshold 0.03 \
#     --n_epoch 200 \
#     --step_size 100 \
#     --patience 5 \
#     --init_cost 1 \
#     --cost_multiplier_up 1.5 \
#     --cost_multiplier_down 2 \
#     --dataset 'stl10' \
#     > logs/stl10/filter_color_wd_$timestamp.log 2>&1 &



# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.01 \
#     --gpu 0 \
#     --batch_size 512 \
#     --ssim_threshold 0.90 \
#     --psnr_threshold 20.0 \
#     --lp_threshold 0.02 \
#     --n_epoch 200 \
#     --step_size 100 \
#     --patience 5 \
#     --init_cost 1 \
#     --cost_multiplier_up 3 \
#     --cost_multiplier_down 1.5 \
#     --dataset 'gtsrb' \
#     > logs/gtsrb/filter_color_wd_$timestamp.log 2>&1 &


######### color loss ablation #########
# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.005 \
#     --gpu 0 \
#     --batch_size 10000 \
#     --ssim_threshold 0.986 \
#     --psnr_threshold 30.4 \
#     --lp_threshold 0.12 \
#     --n_epoch 500 \
#     --step_size 200 \
#     --dataset 'cifar10' \
#     --most_close \
#     > logs/cifar10/most_close_filter_color_wd_$timestamp.log 2>&1 &

# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.005 \
#     --gpu 0 \
#     --batch_size 2048 \
#     --ssim_threshold 0.995 \
#     --psnr_threshold 40.0 \
#     --lp_threshold 0.01 \
#     --n_epoch 200 \
#     --step_size 100 \
#     --dataset 'stl10' \
#     --most_close \
#     > logs/stl10/most_close_filter_color_wd_$timestamp.log 2>&1 &


# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.005 \
#     --gpu 1 \
#     --batch_size 512 \
#     --ssim_threshold 0.95 \
#     --psnr_threshold 25.0 \
#     --lp_threshold 0.01 \
#     --n_epoch 200 \
#     --step_size 100 \
#     --patience 3 \
#     --init_cost 1 \
#     --cost_multiplier_up -0.3 \
#     --cost_multiplier_down -1.1 \
#     --dataset 'stl10' \
#     --ablation \
#     > logs/stl10/ablation_filter_color_wd_$timestamp.log 2>&1 &


### imagenet ###
nohup python main.py \
    --timestamp $timestamp \
    --lr 0.005 \
    --gpu 1 \
    --batch_size 128 \
    --ssim_threshold 0.965 \
    --psnr_threshold 25.0 \
    --lp_threshold 0.1 \
    --n_epoch 150 \
    --step_size 50 \
    --dataset 'imagenet' \
    --most_close \
    > logs/imagenet/most_color_filter_color_wd_$timestamp.log 2>&1 &


### imagenet filter_gtsrb_stl_svhn_ ###
# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.005 \
#     --gpu 4 \
#     --batch_size 38 \
#     --ssim_threshold 0.85 \
#     --psnr_threshold 18.0 \
#     --lp_threshold 0.1 \
#     --n_epoch 100 \
#     --step_size 50 \
#     --patience 5 \
#     --init_cost 3 \
#     --cost_multiplier_up 1.5 \
#     --cost_multiplier_down 1.3 \
#     --dataset 'imagenet_gtsrb_stl10_svhn' \
#     > logs/imagenet/filter_gtsrb_stl_svhn_all_$timestamp.log 2>&1 &

