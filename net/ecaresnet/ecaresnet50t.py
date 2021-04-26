
import math
import jittor as jt
from jittor import nn

class EcaModule(nn.Module):
    """Constructs an ECA module.

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
    """
    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1):
        super(EcaModule, self).__init__()
        assert kernel_size % 2 == 1
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1).sigmoid()
        return x * y.expand_as(x)
    def execute(self, x):
        return self.forward(x)


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_block_rate=0.):
    return [None, None, None, None]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer="eca", aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)

        self.bn2 = norm_layer(width)
        self.act2 = act_layer()
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = EcaModule(outplanes)

        self.act3 = act_layer()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
    
    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x

    def execute(self, x):
        return self.forward(x)

class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = flatten
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return self.pool_type == ''

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x
    def execute(self, x):
        return self.forward(x)


def make_layer(block_fn, inplanes, planes, num_blocks, downsample, stride=1, drop_path_rate=0., first_dilation=1,  dilate=False,avg_down=False):
    block_kwargs = dict(reduce_first=1, dilation=1, drop_block=None, cardinality=1, base_width=64)
    blocks = []
    net_num_blocks = 16
    prev_dilation = first_dilation
    dilation = 1
    down_kernel_size = 1
    if stride != 1 or inplanes != planes * block_fn.expansion:
        down_kwargs = dict(
            in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
            stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=nn.BatchNorm2d)
        downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

    for block_idx in range(num_blocks):
        downsample = downsample if block_idx == 0 else None
        stride = stride if block_idx == 0 else 1
        block_dpr = 0.0  # stochastic depth linear decay rule
        blocks.append(block_fn(
            inplanes, planes, stride, downsample, first_dilation=prev_dilation,
            drop_path=None, **block_kwargs))
        prev_dilation = dilation
        inplanes = planes * block_fn.expansion
    return nn.Sequential(*blocks)


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    output_stride : int, default 32
        Set the output stride of the network, 32, 16, or 8. Typically used in segmentation.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, block, layers, num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64, stem_width=64, stem_type='',
                 output_stride=32, block_reduce_first=1, down_kernel_size=1, avg_down=False,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_rate=0.0, drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None):
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(ResNet, self).__init__()

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer()
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem Pooling
        if aa_layer is not None:
            self.maxpool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=inplanes, stride=2)])
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        
        '''
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        '''

        #block_fn, inplanes, planes, num_blocks, downsample, stride=1,
        self.layer1 = make_layer(Bottleneck, 64, 64, 3, None, 1, 0.0, avg_down=avg_down)
        self.layer2 = make_layer(Bottleneck, 256, 128, 4, None, 2, 0.0, avg_down=avg_down)
        self.layer3 = make_layer(Bottleneck, 512, 256, 6, None, 2, 0.0, avg_down=avg_down)
        self.layer4 = make_layer(Bottleneck, 1024, 512, 3, None, 2, 0.0, avg_down=avg_down)
        
        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True)
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def forward_features(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _forward_impl(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x
    
    def execute(self, x):
        return self._forward_impl(x)


def ecaresnet50t(pretrained=False, **kwargs):
    """Constructs an ECA-ResNet-50-T model.
    Like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'), **kwargs)
    return ResNet(**model_args)



class EcaResNet50t(nn.Module):
    def __init__(self, class_nums=130,pretrain="/home/lipengbo/Download/ecaresnet50t.pkl"):
        base_net = ecaresnet50t()
        base_net.load(pretrain)
        in_channels = base_net.fc.in_features
        new_fc = jt.nn.Linear(in_channels, class_nums, bias=True)
        base_net.fc = new_fc
        self.base_net = base_net

    def execute(self, x):
        x = self.base_net(x)
        return x

# if __name__ == '__main__':
#     import jittor
#     model = EcaResNet50t(class_nums=130)
#     x = jittor.random([10,3,368,368])
#     y = model(x)
#     print(y.shape)


