import os

if not os.path.exists('./log/imagenet/'):
    os.makedirs('./log/imagenet/')


print('Start evaluation')

def evaluate_imagenet(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference, key='clean'):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --batch_size 64 \
            --encoder_usage_info {encoder_usage_info} \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --reference_label {reference_label} \
            --reference_file ./reference/imagenet/{reference}.npz \
            --gpu {gpu} \
            >./log/imagenet/evaluation_{key}_{downstream_dataset}.txt 2>&1 &"

    os.system(cmd)



# evaluate_imagenet(3, 'imagenet', 'stl10', 'output/imagenet/cifar10_backdoored_encoder/model.pth', 9, 'output/imagenet/stl10_backdoored_encoder/trigger_trained.pt', 'truck', 'backdoor')
evaluate_imagenet(3, 'imagenet', 'gtsrb', 'output/imagenet/gtsrb_backdoored_encoder/model.pth', 12, 'output/imagenet/gtsrb_backdoored_encoder/trigger_trained.pt', 'priority', 'backdoor')
# evaluate_imagenet(3, 'imagenet', 'svhn', 'output/imagenet/svhn_backdoored_encoder/model.pth', 1, 'output/imagenet/svhn_backdoored_encoder/trigger_trained.pt', 'one', 'backdoor')


