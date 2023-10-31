import torch.nn as nn
from torch.utils import checkpoint as cp
from functools import partial
from module import CVModule
from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['make_layer',
           'ResNetEncoder', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def freeze_params(module):
    for name, p in module.named_parameters():
        p.requires_grad = False
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def freeze_modules(module, specific_class=None):
    for m in module.modules():
        if specific_class is not None:
            if not isinstance(m, specific_class):
                continue
        freeze_params(m)


def make_layer(block, in_channel, basic_out_channel, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or in_channel != basic_out_channel * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channel, basic_out_channel * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(basic_out_channel * block.expansion),
        )

    layers = []
    layers.append(block(in_channel, basic_out_channel, stride, dilation, downsample))
    in_channel = basic_out_channel * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channel, basic_out_channel, dilation=dilation))

    return nn.Sequential(*layers)


class ResNetEncoder(CVModule):
    def __init__(self,
                 config):
        super(ResNetEncoder, self).__init__(config)
        if all([self.config.output_stride != 16,
                self.config.output_stride != 32,
                self.config.output_stride != 8]):
            raise ValueError('output_stride must be 8, 16 or 32.')

        self.resnet = resnet50(pretrained=self.config.pretrained,
                               norm_layer=self.config.norm_layer)
        self.resnet._modules.pop('fc')
        if not self.config.batchnorm_trainable:
            self._frozen_res_bn()

        self._freeze_at(at=self.config.freeze_at)

        if self.config.output_stride == 16:
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        elif self.config.output_stride == 8:
            self.resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))

    @property
    def layer1(self):
        return self.resnet.layer1

    @layer1.setter
    def layer1(self, value):
        del self.resnet.layer1
        self.resnet.layer1 = value

    @property
    def layer2(self):
        return self.resnet.layer2

    @layer2.setter
    def layer2(self, value):
        del self.resnet.layer2
        self.resnet.layer2 = value

    @property
    def layer3(self):
        return self.resnet.layer3

    @layer3.setter
    def layer3(self, value):
        del self.resnet.layer3
        self.resnet.layer3 = value

    @property
    def layer4(self):
        return self.resnet.layer4

    @layer4.setter
    def layer4(self, value):
        del self.resnet.layer4
        self.resnet.layer4 = value

    def _frozen_res_bn(self):
        freeze_modules(self.resnet, nn.modules.batchnorm._BatchNorm)
        for m in self.resnet.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def _freeze_at(self, at=2):
        if at >= 1:
            freeze_params(self.resnet.conv1)
            freeze_params(self.resnet.bn1)
        if at >= 2:
            freeze_params(self.resnet.layer1)
        if at >= 3:
            freeze_params(self.resnet.layer2)
        if at >= 4:
            freeze_params(self.resnet.layer3)
        if at >= 5:
            freeze_params(self.resnet.layer4)

    @staticmethod
    def get_function(module):
        def _function(x):
            y = module(x)
            return y

        return _function

    def forward(self, inputs):
        x = inputs
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # os 4, #layers/outdim: 18,34/64; 50,101,152/256
        if self.config.with_cp[0] and x.requires_grad:
            c2 = cp.checkpoint(self.get_function(self.resnet.layer1), x)
        else:
            c2 = self.resnet.layer1(x)
        # os 8, #layers/outdim: 18,34/128; 50,101,152/512
        if self.config.with_cp[1] and c2.requires_grad:
            c3 = cp.checkpoint(self.get_function(self.resnet.layer2), c2)
        else:
            c3 = self.resnet.layer2(c2)
        # os 16, #layers/outdim: 18,34/256; 50,101,152/1024
        if self.config.with_cp[2] and c3.requires_grad:
            c4 = cp.checkpoint(self.get_function(self.resnet.layer3), c3)
        else:
            c4 = self.resnet.layer3(c3)
        # os 32, #layers/outdim: 18,34/512; 50,101,152/2048
        if self.config.include_conv5:
            if self.config.with_cp[3] and c4.requires_grad:
                c5 = cp.checkpoint(self.get_function(self.resnet.layer4), c4)
            else:
                c5 = self.resnet.layer4(c4)
            return [c2, c3, c4, c5]

        return [c2, c3, c4]

    def set_defalut_config(self):
        self.config.update(dict(
            include_conv5=True,
            batchnorm_trainable=True,
            pretrained=False,
            freeze_at=0,
            # 16 or 32
            output_stride=32,
            with_cp=(False, False, False, False),
            norm_layer=nn.BatchNorm2d,
        ))

    def train(self, mode=True):
        super(ResNetEncoder, self).train(mode)
        self._freeze_at(self.config.freeze_at)
        if mode and not self.config.batchnorm_trainable:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def _nostride_dilate(self, m, dilate):
        # ref:
        # https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/1235deb1d68a8f3ef87d639b95b2b8e3607eea4c/models/models.py#L256
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
