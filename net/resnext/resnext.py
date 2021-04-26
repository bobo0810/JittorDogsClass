from jittor.models import Resnext101_32x8d,Resnext50_32x4d
import jittor.nn as nn
# from net.resnext.resnext50 import Resnext50_32x4d  #加入DyRelu

class ResNeXt50(nn.Module):
    def __init__(self, class_nums=130):
        base_net = Resnext50_32x4d(pretrained=True)
        in_channels = base_net.fc.in_features
        new_fc = nn.Linear(in_channels, class_nums, bias=True)
        base_net.fc=new_fc
        self.base_net=base_net
    def execute(self, x):
        x = self.base_net(x)
        return x

class ResNeXt101(nn.Module):
    def __init__(self, class_nums=130):
        base_net = Resnext101_32x8d(pretrained=True)
        in_channels = base_net.fc.in_features
        new_fc = nn.Linear(in_channels, class_nums, bias=True)
        base_net.fc=new_fc
        self.base_net=base_net
    def execute(self, x):
        x = self.base_net(x)
        return x
# if __name__ == '__main__':
#     import jittor
#     model = ResNeXt50(class_nums=130)
#     x = jittor.random([10,3,368,368])
#     y = model(x)
#     print(y.shape)