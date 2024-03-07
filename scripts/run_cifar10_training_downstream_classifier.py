import os

# if not os.path.exists('./log/cifar10'):
#     os.makedirs('./log/cifar10')

print('Start evaluation')
def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference_file, key='clean'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # out = f'./log/{encoder_usage_info}'
    # os.makedirs(out, exist_ok=True)
    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            --reference_label {reference_label} \
            --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
            --gpu {gpu} \
            > ./log/{encoder_usage_info}/evaluation_{key}_{encoder_usage_info}_{downstream_dataset}.log 2>&1 &"


    os.system(cmd)





run_eval(1, 'stl10', 'cifar10', ' ./output/stl10/cifar10_backdoored_encoder/model.pth', 0, './output/stl10/cifar10_backdoored_encoder/trigger_trained.pt', 'airplane', 'backdoor')
# run_eval(2, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/model.pth', 12, './output/stl10/gtsrb_backdoored_encoder/trigger_trained.pt', 'priority', 'backdoor')
# run_eval(2, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/model.pth', 1, './output/stl10/svhn_backdoored_encoder/trigger_trained.pt', 'one', 'backdoor')



