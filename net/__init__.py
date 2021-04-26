import sys
import yaml
from .resnext.resnext import ResNeXt50,ResNeXt101
from .ecaresnet.ecaresnet50t import EcaResNet50t
import jittor.nn as nn
import os
cur_path = os.path.abspath(os.path.dirname(__file__))

class BackboneFactory(nn.Module):
    '''
    主干网络统一入口
    '''
    def __init__(self, backbone_type,backbone_conf_file='backbone_conf.yaml'):
        super(BackboneFactory, self).__init__()
        self.model = self.get_backbone(backbone_type,backbone_conf_file)

    def execute(self, x):
        return self.model(x)

    @staticmethod
    def get_backbone(backbone_type,backbone_conf_file):
        '''
        初始化主干网络
        '''
        # 读取yaml配置
        file = open(os.path.join(cur_path,backbone_conf_file), 'r',encoding="utf-8")
        backbone_conf = yaml.load(file, Loader=yaml.FullLoader)
        conf = backbone_conf[backbone_type]


        class_nums = conf['class_nums']

        if backbone_type=="ResNeXt50":
            backbone = ResNeXt50(class_nums=class_nums)
        elif backbone_type=="ResNeXt101":
            backbone = ResNeXt101(class_nums=class_nums)
        elif backbone_type == "EcaResNet50t":
            backbone = EcaResNet50t(class_nums=class_nums)
        else:
            raise NotImplementedError
        return backbone

