import os

cifar10_results_dir = './output/cifar10/clean_encoder/'
stl10_results_dir = './output/stl10/clean_encoder/'
gtsrb_results_dir = './output/gtsrb/clean_encoder/'
svhn_results_dir = './output/svhn/clean_encoder/'

imagenet_results_dir = './output/tiny_imagenet/clean_encoder/'

if not os.path.exists('./log/clean_encoder'):
    os.makedirs('./log/clean_encoder')
if not os.path.exists(cifar10_results_dir):
    os.makedirs(cifar10_results_dir)
if not os.path.exists(stl10_results_dir):
    os.makedirs(stl10_results_dir)
if not os.path.exists(gtsrb_results_dir):
    os.makedirs(gtsrb_results_dir)
if not os.path.exists(svhn_results_dir):
    os.makedirs(svhn_results_dir)

cmd = f"nohup python3 -u pretraining_encoder.py --pretraining_dataset cifar10 --gpu 2 --results_dir {cifar10_results_dir} > ./log/clean_encoder/cifar10.log &"
os.system(cmd)

# cmd = f"nohup python3 -u pretraining_encoder.py --pretraining_dataset stl10 --gpu 0 --results_dir {stl10_results_dir} > ./log/clean_encoder/stl10.log &"
# os.system(cmd)

