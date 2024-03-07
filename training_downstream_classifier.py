import os
import argparse
import random

import torchvision
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataset_evaluation
from models import get_encoder_architecture_usage
from evaluation import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature
from datetime import datetime
import pickle

if __name__ == '__main__':
    # print(torch.cuda.is_available())
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    now = datetime.now()
    print("当前时间：", now.strftime("%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser(description='Evaluate the clean or backdoored encoders')
    parser.add_argument('--dataset', default='cifar10', type=str, help='downstream dataset')
    parser.add_argument('--reference_label', default=-1, type=int, help='target class in the target downstream task')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger file (default: none)')
    parser.add_argument('--encoder_usage_info', default='', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='', type=str, help='path to the image encoder')

    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--nn_epochs', default=500, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    ## note that the reference_file is not needed to train a downstream classifier
    parser.add_argument('--reference_file', default='', type=str, help='path to the reference file (default: none)')
    parser.add_argument('--noise', type=str)

    args = parser.parse_args()


    # random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


    assert args.reference_label >= 0, 'Enter the correct target class'


    args.data_dir = f'./data/{args.dataset}/'
    target_dataset, train_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(args)


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                   pin_memory=True)
    test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                      pin_memory=True)

    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    num_of_classes = len(train_data.classes)

    model = get_encoder_architecture_usage(args).cuda()

    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder)
        args_v = checkpoint.get('args', None)
        loss = checkpoint.get('loss', None)
        if args_v:
            print(args_v)
        if loss:
            print(loss)

        if args.encoder_usage_info in ['CLIP', 'imagenet'] and 'clean' in args.encoder:
            model.visual.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

    print('Predicting features')
    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        feature_bank_training, label_bank_training = predict_feature(args,model.visual, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(args,model.visual, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(args,model.visual, test_loader_backdoor,'backdoor')
        feature_bank_target, label_bank_target = predict_feature(args,model.visual, target_loader)
    else:
        feature_bank_training, label_bank_training = predict_feature(args,model.f, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(args,model.f, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(args,model.f, test_loader_backdoor,'backdoor')
        feature_bank_target, label_bank_target = predict_feature(args,model.f, target_loader)

    print(feature_bank_training.shape[1])
    feature_banks = {
        'args':args_v,
        'training': feature_bank_training,
        'testing': feature_bank_testing,
        'backdoor': feature_bank_backdoor,
        'target': feature_bank_target
    }


    # out = os.path.join(os.path.dirname(args.encoder), 'feature_banks.pkl')
    # with open(out, 'wb') as f:
    #     pickle.dump(feature_banks, f)
    # print('feature banks saved to {}'.format(out))

    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size)

    input_size = feature_bank_training.shape[1]

    criterion = nn.CrossEntropyLoss()

    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(1, args.nn_epochs + 1):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)
        if 'clean' in args.encoder:
            net_test(net, nn_test_loader, epoch, criterion, 'Clean Accuracy (CA)')
            net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate-Baseline (ASR-B)')
        else:
            net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
            net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')

    save_path= os.path.join(os.path.dirname(args.encoder), 'classifier.pkl')

    torch.save(net.state_dict(), save_path)
    print("save classifier to ", save_path)
    now = datetime.now()
    print("当前时间：", now.strftime("%Y-%m-%d %H:%M:%S"))
