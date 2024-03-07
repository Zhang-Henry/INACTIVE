timestamp=$(date +"%Y-%m-%d-%H-%M-%S")


nohup python main.py \
    --timestamp $timestamp \
    --lr 0.01 \
    --gpu 5 \
    --batch_size 2048 \
    --ssim_threshold 0.80 \
    --psnr_threshold 13.0 \
    --lp_threshold 0.1 \
    --n_epoch 200 \
    --step_size 100 \
    --patience 5 \
    --init_cost 1 \
    --cost_multiplier_up 1.5 \
    --cost_multiplier_down 2 \
    --dataset 'cifar10' \
    > logs/cifar10/trigger_pretrain_$timestamp.log 2>&1 &





