from utils.sam import SAM
import math
import random
import socket
from jittor.optim import SGD
from dataset.TrainSet import BalancedBatchSampler_TsinghuaDog,TsinghuaDog
import os
import numpy as np
import jittor as jt
from visualdl import LogWriter
import jittor
cur_path = os.path.abspath(os.path.dirname(__file__))
def get_host_ip(complete=False):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    if complete:
        return ip # 192.168.1.66
    else:
        return ip.split('.')[-1] # 66

def init_seed(seed: int):
    '''
    设置随机种子
    '''
    random.seed(seed)
    jt.set_global_seed(seed)

class LR_Schduler:
    '''
    学习率调度器
    ReduceLROnPlateau:按需调整学习率
    Warmup_CosLR: 带预热的余弦退火学习率
    '''
    def __init__(self,optimizer_cfg, optimizer ):
        self.optimizer_cfg = optimizer_cfg
        self.type=optimizer_cfg['schduler']['type']

        if self.type == 'Warmup_CosLR':
            scheduler = self.warmup_cosine_schduler(optimizer_cfg, optimizer)
        else:
            raise NotImplementedError
        self.scheduler=scheduler
    def step(self):
        '''
        更新学习率
        '''
        if self.type == 'Warmup_CosLR':
            self.scheduler.step()
        else:
            raise NotImplementedError
    def get_lr(self):
        if self.type == 'Warmup_CosLR':
            lr=self.scheduler.get_lr()
        else:
            raise NotImplementedError
        return lr
    def warmup_cosine_schduler(self,optimizer_cfg, optimizer):
        '''
        带预热的余弦退火策略
        '''
        warm_up_epochs = optimizer_cfg['schduler']['warmup']
        total_epochs = optimizer_cfg['epochs']
        warmup_cosine_lr = lambda epoch: 0.5 * (
                    math.cos((epoch - warm_up_epochs) / (total_epochs - warm_up_epochs) * math.pi) + 1) \
            if warm_up_epochs == 0 or epoch > warm_up_epochs else epoch / warm_up_epochs
        scheduler = jt.optim.LambdaLR(optimizer,lr_lambda=warmup_cosine_lr)
        return scheduler

def save_model(backbone, cfg,epoch):
    backbone.save('%s_%d.pkl' % (cfg['checkpoint_dir'], epoch))

def tb_log(cfg,localtime):
    '''
    初始化 可视化工具
    '''
    logdir = os.path.join(cur_path+'/../tb_log/',localtime + '_ip' + str(get_host_ip()))
    writer=None
    if jt.rank == 0:
        writer = LogWriter(logdir=logdir)
        writer.add_text('config', str(cfg))
        print('\n\ncfg is '+str(cfg)+"\n\n")
        print('log file in ' + logdir)
    return writer

def load_dataset(cfg,mode):
    assert mode in ['train','eval']

    # 训练集
    if mode=='train':
        dataloader = BalancedBatchSampler_TsinghuaDog(n_classes=cfg['train']['n_classes'],
                                                     n_samples=cfg['train']['n_samples'],
                                                     img_size=cfg['train']['img_size'],
                                                     mode=mode,shuffle=True, num_workers=3,
                                                    crop_type=cfg['train']['crop_type'])

    else:
        dataloader = TsinghuaDog(img_size=cfg['train']['img_size'],
                                 mode=mode, shuffle=False,
                                 batch_size=cfg['train']['n_classes']*cfg['train']['n_samples'],
                                 num_workers=2,crop_type=cfg['train']['crop_type'])
    return dataloader

def save_dir(cfg,localtime):
    '''
    新建模型保存路径
    '''
    cfg['checkpoint_dir'] = os.path.join(cur_path + '/../CheckPoint/', localtime + '/model')
    if not os.path.exists(os.path.dirname(cfg['checkpoint_dir'])):
        os.makedirs(os.path.dirname(cfg['checkpoint_dir'] ))
        print('save model in '+ os.path.dirname(cfg['checkpoint_dir'] ))

def optimizer_init(cfg,backbone):
    '''
    初始化优化器
    '''
    # 优化器
    if cfg['optimizer']['type']=="SGD":
        optimizer = SGD(backbone.parameters(),lr=cfg['optimizer']['lr'],weight_decay=0.0005,momentum=0.9)
    elif cfg['optimizer']['type']=="SAM":
        base_optimizer = SGD(backbone.parameters(),lr=cfg['optimizer']['lr'],weight_decay=0.0005,momentum=0.9)
        optimizer = SAM(base_optimizer, lr=cfg['optimizer']['lr'], rho=0.05)
    else:
        raise NotImplementedError
    return optimizer


def load_model_weights(backbone,net_cfg,writer):
    '''
    加载主干网络权重
    '''
    if net_cfg['pretrain']:
        # 第一种访问
        # backbone.load(net_cfg['pretrain'])
        # print('第一种加载权重方式')

        # 第二种访问
        params_dict=jittor.load(net_cfg['pretrain'])
        backbone.load_state_dict(params_dict)
        # print('第二种加载权重方式')


        if jt.rank == 0:
            writer.add_text('config', " loading  based on {}".format(net_cfg['pretrain']))
    else:
        if jt.rank == 0:
            writer.add_text('config', " loading based on None")
    return backbone


@jittor.no_grad()
def eval_acc(params):
    eval_dataloader, backbone=params
    backbone.eval()  # 固定BN和DropOut
    total_acc = 0
    total_num = 0
    for batch_idx, (images, labels) in enumerate(eval_dataloader):

        output = backbone(images)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
    acc = total_acc / total_num
    return acc



