import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_pruning as tp

"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.skip_add = nn.quantized.FloatFunctional()
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        m = self.skip_add.add_scalar(x, 3)
        return self.relu(self.skip_mul.mul_scalar(m, 1 / 6))


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_mul.mul(x, self.sigmoid(x))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.skip_mul = nn.quantized.FloatFunctional()
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return self.skip_mul.mul(x, y)


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        self.skip_add = nn.quantized.FloatFunctional()

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self) -> None:
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) is nn.Conv2d:
                torch.quantization.fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, mnist=False, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        if mnist:
            in_ch = 1
        else:
            in_ch = 3
        layers = [conv_3x3_bn(in_ch, input_channel, 2)]

        self.strides = list()
        self.strides.append(2)
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            self.strides.append(s)
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
            mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(mnist=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', mnist=mnist)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


def prune_tp(model, agg=False):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 192, 640))

    def prune_conv(conv, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    if agg:
        probs = [0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6]
        enc_ch = np.array([32, 24, 32, 64, 1280])
        dec_ch = np.array([6, 12, 24, 32, 64])
    else:
        probs = [0.3, 0.3, 0.3,
                 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        enc_ch = np.array([16, 24, 40, 56, 576])
        dec_ch = np.array([6, 12, 24, 40, 56])

    layer_list = ['encoder.features.7.conv.0', 'encoder.features.7.conv.3', 'encoder.features.7.conv.7',
                  'encoder.features.12.conv.0', 'encoder.features.12.conv.3', 'encoder.features.12.conv.7',
                  'encoder.features.13.conv.0', 'encoder.features.13.conv.3', 'encoder.features.13.conv.7',
                  'encoder.features.14.conv.0', 'encoder.features.14.conv.3', 'encoder.features.14.conv.7',
                  'encoder.features.15.conv.0', 'encoder.features.15.conv.3', 'encoder.features.15.conv.7',
                  'encoder.conv.0'
                  ]
    blk_id = 0
    for name, module in model.named_modules():
        if name in layer_list and isinstance(module, nn.Conv2d):
            print(name)
            prune_conv(module, probs[blk_id])
            blk_id += 1
    return model, enc_ch, dec_ch


class MobileNetv3Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetv3Encoder, self).__init__()

        self.num_ch_enc = np.array([16, 24, 40, 80, 960])
        self.num_ch_dec = np.array([6, 12, 24, 40, 80])
        self.prune = prune_tp

        self.encoder = mobilenetv3_large()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        if pretrained:
            dir = os.path.dirname(__file__)
            state_dict = torch.load(os.path.join(dir, 'mobilenetv3-large-1cd25616.pth'))
            self.encoder.load_state_dict(state_dict, strict=True)

    def forward(self, input_image):
        self.features = []
        out = (input_image - 0.45) / 0.225
        out = self.quant(out)
        for i in range(len(self.encoder.features)):
            out = self.encoder.features[i](out)
            if self.encoder.strides[i] > 1:
                self.features.append(out)
        self.features = self.features[:-1]
        self.features.append(self.encoder.conv(out))
        for i in range(len(self.features)):
            self.features[i] = self.dequant(self.features[i])
        return self.features

    def fuse_model(self) -> None:
        torch.quantization.fuse_modules(self, ['encoder.features.0.0', 'encoder.features.0.1'], inplace=True)
        for i, module in self.named_modules():
            if isinstance(module, InvertedResidual):
                module.fuse_model()
        torch.quantization.fuse_modules(self, ['encoder.conv.0', 'encoder.conv.1'], inplace=True)


class ExpModel_Mobv3(MobileNetv3Encoder):
    def __init__(self, pretrained=False, mnist=False):
        super(ExpModel_Mobv3, self).__init__()

        self.encoder = mobilenetv3_large()

    def forward(self, x):
        out = self.quant(x)
        out = self.encoder(out)
        out = self.dequant(out)
        return out


if __name__ == '__main__':
    net = ExpModel_Mobv3()
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    # net, enc, dec = net.prune(net)
    net.eval()
    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(net, inplace=True)
    _dummy_input_data = torch.rand(1, 3, 192, 640)
    x = net(_dummy_input_data)
    torch.quantization.convert(net, inplace=True)
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

