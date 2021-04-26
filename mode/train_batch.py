from .train_init import train_init
from utils.tools import save_model,eval_acc
from utils.focalloss import FocalLoss
import jittor as jt
import numpy as np
import random

def train(cfg):

    init_param=train_init(cfg)
    optimizer = init_param['optimizer']
    lr_scheduler = init_param['lr_scheduler']
    writer = init_param['writer']
    cfg = init_param['cfg']
    backbone = init_param['backbone']
    train_dataloader = init_param['train_dataloader']
    eval_dataloader = init_param['eval_dataloader']

    if 'FocalLoss' in cfg['train']['loss'] :
        criterion = FocalLoss()
    elif 'CrossEntropy' in cfg['train']['loss']:
        criterion = jt.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # 开始训练
    for epoch in range(0, cfg['optimizer']['epochs']):
        # warm-up导致epoch==0的lr=0
        if cfg['optimizer']['schduler']['type'] in 'Warmup_CosLR' and cfg['optimizer']['schduler']['warmup'] != 0 and epoch == 0:
            lr_scheduler.step()
            continue
        if jt.rank == 0:
            print('start epoch {}/{}.'.format(epoch, cfg['optimizer']['epochs']))
            writer.add_scalar(tag='learning_rate',step=epoch,value=lr_scheduler.get_lr()[0])

        # 优化器梯度清零
        optimizer.zero_grad()
        backbone.train()

        for batch_idx, (input_imgs, labels) in enumerate(train_dataloader):

            # 打乱BN
            batch_size= list(range(len(labels)))
            random.shuffle(batch_size)
            input_imgs = input_imgs[batch_size]
            labels=labels[batch_size]

            if cfg['optimizer']['type'] == "SGD":
                output=backbone(input_imgs)
                loss = criterion(output, labels)
                optimizer.step(loss)
            elif cfg['optimizer']['type'] == "SAM":
                output = backbone(input_imgs)
                loss = criterion(output, labels)
                optimizer.first_step(loss)

                output = backbone(input_imgs)
                loss = criterion(output, labels)
                optimizer.second_step(loss)
            else:
                raise NotImplementedError

            if batch_idx % 100 == 0:
                pred = np.argmax(output.data, axis=1)
                acc = np.mean(pred == labels.data) * 100

                iter_num = int(batch_idx + epoch * len(train_dataloader))
                if jt.rank == 0:
                    writer.add_scalar(tag='Train/loss', step=iter_num,value=loss.data[0])
                    writer.add_scalar(tag='Train/batch_acc', step=iter_num,value=acc)


        acc = eval_acc([eval_dataloader, backbone]) # 评估
        if jt.rank == 0:
            print("acc" + str(acc))
            writer.add_scalar(tag='Eval/Acc', step=epoch, value=acc)
            save_model(backbone,cfg,epoch)
        lr_scheduler.step()
