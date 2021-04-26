from utils.tools import init_seed,load_dataset,tb_log,save_dir,optimizer_init,LR_Schduler,load_model_weights
from net import BackboneFactory
import time


def train_init(cfg):
    init_seed(227) # 随机种子

    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(" ","_")
    writer = tb_log(cfg,localtime) # 记录log
    save_dir(cfg,localtime)  # 模型保存路径

    # 初始化网络和分类器
    backbone = BackboneFactory(cfg['net']['name'])

    # 优化器
    optimizer = optimizer_init(cfg,backbone)

    # 加载权重
    backbone = load_model_weights(backbone,cfg['net'],writer)

    # 转为训练模式
    backbone.train()


    # 学习率调度器
    lr_scheduler= LR_Schduler(cfg['optimizer'],optimizer=optimizer)

    # 训练集/验证集
    train_dataloader = load_dataset(cfg,mode='train')
    eval_dataloader = load_dataset(cfg,mode='eval')

    init_param = {'optimizer': optimizer,
                  'lr_scheduler': lr_scheduler, 'writer': writer,
                  'backbone': backbone,'cfg': cfg,
                  'train_dataloader': train_dataloader, 'eval_dataloader': eval_dataloader,
                  }
    return init_param