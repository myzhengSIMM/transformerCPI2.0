# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/29 14:21
@author: LiFan Chen
@Filename: main_amp.py
@Software: PyCharm
"""

import torch
import random
import os
from model_amp import *
import timeit
# from utils import *
from torch.cuda.amp import GradScaler
def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy')]

def check_dictory(path):
    if  not os.path.exists(path):#如果路径不存在
        os.makedirs(path)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    SEED = 8686
    random.seed(SEED)
    torch.manual_seed(SEED)
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    """Load preprocessed data."""

    DATASET = "GPCR"
    print(DATASET + ' loading...')
    dir_input = ('data/processed/' + DATASET + '/')
    #dir_input = ('data/processed/')
    compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    dataset_train = list(zip(compounds, adjacencies, proteins, interactions))

    DATASET = "GPCR"
    print(DATASET + ' loading...')
    dir_input = ('data/processed/' + DATASET + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    dataset_dev = list(zip(compounds, adjacencies, proteins, interactions))

    DATASET = "GPCR"
    print(DATASET + ' loading...')
    dir_input = ('data/processed/' + DATASET + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    dataset_test = list(zip(compounds, adjacencies, proteins, interactions))

    """ create model ,trainer and tester """
    n_layers = 3
    dropout = 0.2
    batch = 10
    lr = 1e-5
    weight_decay = 1e-3
    decay_interval = 10
    lr_decay = 0.5
    iteration = 60

    pretrain = torch.load('Bert.pkl')
    pretrain.to(device)
    for param in pretrain.parameters():
        param.requires_grad = False

    pretrain.eval()
    encoder = Encoder(pretrain,n_layers,device)
    decoder = Decoder(n_layers, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'output/result/AUCs--lr=1e-5,weight_decay=1e-3,dropout=0.2,batch=%s,amp.txt' % batch
    file_model = 'output/model/' + 'lr=1e-5,weight_decay=1e-3,dropout=0.2,batch=%s,amp.pt' % batch
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'PRC_dev\tAUC_test\tPRC_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    max_AUC_dev = 0
    scaler = GradScaler()
    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, device,scaler)
        AUC_dev, PRC_dev = tester.test(dataset_dev)
        AUC_test, PRC_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev, PRC_dev, AUC_test, PRC_test] 
        tester.save_AUCs(AUCs, file_AUCs)
        if AUC_dev > max_AUC_dev:
            tester.save_model(model, file_model)
            max_AUC_dev = AUC_dev
        print('\t'.join(map(str, AUCs)))
