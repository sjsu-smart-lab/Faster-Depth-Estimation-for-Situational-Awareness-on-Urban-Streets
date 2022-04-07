import math
import torch
from torch import nn
import numpy as np
import torch_pruning as tp

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'efficientnet_b0': 'https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1',
    'efficientnet_b1': 'https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1',
    'efficientnet_b2': 'https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1',
    'efficientnet_b3': 'https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1',
    'efficientnet_b4': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    'efficientnet_b5': 'https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1',
    'efficientnet_b6': None,
    'efficientnet_b7': None,
}

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_mul.mul(x, torch.sigmoid(x))


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_mul.mul(x, self.se(x))


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)
        self.skip_add = nn.quantized.FloatFunctional()
        self.skip_mul = nn.quantized.FloatFunctional()

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return self.skip_mul.mul(x.div(keep_prob), binary_tensor)

    def forward(self, x):
        if self.use_residual:
            return self.skip_add.add(x, self._drop_connect(self.conv(x)))
        else:
            return self.conv(x)

    def fuse_model(self) -> None:
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) is nn.Conv2d:
                torch.quantization.fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


class EfficientNet(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000, mnist=False):
        super(EfficientNet, self).__init__()

        # yapf: disable
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        # yapf: enable

        out_channels = _round_filters(32, width_mult)
        if mnist:
            in_ch = 1
        else:
            in_ch = 3
        features = [ConvBNReLU(in_ch, out_channels, 3, stride=2)]

        self.strides = list()
        self.strides.append(2)
        in_channels = out_channels
        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                self.strides.append(stride)
                in_channels = out_channels

        last_channels = _round_filters(1280, width_mult)
        features += [ConvBNReLU(in_channels, last_channels, 1)]
        self.strides.append(1)
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def _efficientnet(arch, pretrained, progress, **kwargs):
    width_mult, depth_mult, _, dropout_rate = params[arch]
    model = EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b0', pretrained, progress, **kwargs)


def efficientnet_b1(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b1', pretrained, progress, **kwargs)


def efficientnet_b2(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b2', pretrained, progress, **kwargs)


def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b3', pretrained, progress, **kwargs)


def efficientnet_b4(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b4', pretrained, progress, **kwargs)


def efficientnet_b5(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b5', pretrained, progress, **kwargs)


def efficientnet_b6(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b6', pretrained, progress, **kwargs)


def efficientnet_b7(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b7', pretrained, progress, **kwargs)


def prune_tp(model, agg=False):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 192, 640))

    def prune_conv(conv, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    if agg:
        probs = [0.2, 0.2, 0.2,
                 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        enc_ch = np.array([32, 24, 32, 52, 896])
        dec_ch = np.array([6, 12, 24, 32, 52])
    else:
        probs = [0.1, 0.1, 0.1,
                 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        enc_ch = np.array([32, 24, 32, 58, 1024])
        dec_ch = np.array([6, 12, 24, 32, 58])

    layer_list = ['encoder.features.6.conv.0.1', 'encoder.features.6.conv.1.1', 'encoder.features.6.conv.3',
                  'encoder.features.13.conv.0.1', 'encoder.features.13.conv.1.1', 'encoder.features.13.conv.3',
                  'encoder.features.14.conv.0.1', 'encoder.features.14.conv.1.1', 'encoder.features.14.conv.3',
                  'encoder.features.15.conv.0.1', 'encoder.features.15.conv.1.1', 'encoder.features.15.conv.3',
                  'encoder.features.16.conv.0.1', 'encoder.features.16.conv.1.1', 'encoder.features.16.conv.3',
                  'encoder.features.17.1'
                  ]
    blk_id = 0
    for name, module in model.named_modules():
        if name in layer_list and isinstance(module, nn.Conv2d):
            prune_conv(module, probs[blk_id])
            blk_id += 1
    return model, enc_ch, dec_ch


class EfficientEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super(EfficientEncoder, self).__init__()

        self.num_ch_enc = np.array([32, 24, 40, 80, 1280])
        self.num_ch_dec = np.array([6, 12, 24, 40, 80])
        self.prune = prune_tp

        self.encoder = efficientnet_b0(pretrained=pretrained)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input_image):
        self.features = []
        out = (input_image - 0.45) / 0.225
        out = self.quant(out)
        for i in range(len(self.encoder.features)):
            out = self.encoder.features[i](out)
            if self.encoder.strides[i] > 1:
                self.features.append(out)
        self.features = self.features[:-1]
        self.features.append(out)
        for i in range(len(self.features)):
            self.features[i] = self.dequant(self.features[i])
        return self.features

    def fuse_model(self) -> None:
        for m in self.modules():
            if type(m) is ConvBNReLU:
                torch.quantization.fuse_modules(m, ["1", "2"], inplace=True)
            if type(m) is MBConvBlock:
                m.fuse_model()


class ExpModel_Effnet(EfficientEncoder):
    def __init__(self, pretrained=False, progress=True, mnist=False):
        super(ExpModel_Effnet, self).__init__()

        self.encoder = efficientnet_b0(num_classes=10, mnist=mnist)

    def forward(self, input_image):
        out = self.quant(input_image)
        out = self.encoder(out)
        out = self.dequant(out)
        return out


if __name__ == '__main__':
    net = EfficientEncoder(pretrained=True)
    print(net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net, enc, dec = net.prune(net)
    # net.eval()
    # net.fuse_model()
    # net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # torch.quantization.prepare(net, inplace=True)
    _dummy_input_data = torch.rand(1, 3, 192, 640)
    x = net(_dummy_input_data)
    # torch.quantization.convert(net, inplace=True)
    print(net)
    print(enc)
    print(dec)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

