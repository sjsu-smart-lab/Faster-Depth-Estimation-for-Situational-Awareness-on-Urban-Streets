from torch import nn
import numpy as np
import torch
import prune
import torch_pruning as tp

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self) -> None:
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) is nn.Conv2d:
                torch.quantization.fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 mnist=False):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        if mnist:
            in_ch = 1
        else:
            in_ch = 3
        features = [ConvBNReLU(in_ch, input_channel, stride=2)]

        self.strides = list()
        self.strides.append(2)
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                self.strides.append(stride)
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.strides.append(1)
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


# def prune_model(model, agg=False, independent_prune_flag=False):
#     if agg:
#         prune_channels = [96, 96, 96, 96, 768]
#     else:
#         prune_channels = [26, 26, 26, 640, 640, 640, 640]
#     prune_layers = ['conv19', 'conv20', 'conv21']
#     # model = prune_step(model, InvertedResidual, ['conv18', 'conv19', 'conv20', 'conv21', 'conv22'], block_prune_probs)
#     count = 0  # count for indexing 'prune_channels'
#     conv_count = 1  # conv count for 'indexing_prune_layers'
#     dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
#     residue = None  # residue is need to prune by 'independent strategy'
#     for module_name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv2d):
#             if dim == 1:
#                 print('hi')
#                 new_, residue = prune.get_new_conv(module, dim, channel_index,
#                                                    independent_prune_flag)
#                 module = new_
#                 dim ^= 1
#
#             if 'conv%d' % conv_count in prune_layers:
#                 print('conv')
#                 print(module_name)
#                 channel_index = prune.get_channel_index(module.weight.data, prune_channels[count],
#                                                         residue)
#                 new_ = prune.get_new_conv(module, dim, channel_index, independent_prune_flag)
#                 module = new_
#                 dim ^= 1
#                 count += 1
#             else:
#                 residue = None
#             conv_count += 1
#
#         if dim == 1 and isinstance(module, torch.nn.BatchNorm2d):
#             print('bn')
#             new_ = prune.get_new_norm(module, channel_index)
#             module = new_
#     return model


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
                 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                 0.5]
        enc_ch = np.array([32, 20, 17, 45, 640])
        dec_ch = np.array([5, 10, 20, 17, 45])
    else:
        probs = [0.1, 0.1, 0.1,
                 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                 0.3]
        enc_ch = np.array([32, 22, 21, 52, 896])
        dec_ch = np.array([6, 11, 22, 21, 52])

    layer_list = ['encoder.features.2.conv.0.0', 'encoder.features.2.conv.1.0', 'encoder.features.2.conv.2',
                  'encoder.features.4.conv.0.0', 'encoder.features.4.conv.1.0', 'encoder.features.4.conv.2',
                  'encoder.features.6.conv.0.0', 'encoder.features.6.conv.1.0', 'encoder.features.6.conv.2',
                  'encoder.features.7.conv.0.0', 'encoder.features.7.conv.1.0', 'encoder.features.7.conv.2',
                  'encoder.features.13.conv.0.0', 'encoder.features.13.conv.1.0', 'encoder.features.13.conv.2',
                  'encoder.features.14.conv.0.0', 'encoder.features.14.conv.1.0', 'encoder.features.14.conv.2',
                  'encoder.features.15.conv.0.0', 'encoder.features.15.conv.1.0', 'encoder.features.15.conv.2',
                  'encoder.features.16.conv.0.0', 'encoder.features.16.conv.1.0', 'encoder.features.16.conv.2',
                  'encoder.features.17.conv.0.0', 'encoder.features.17.conv.1.0', 'encoder.features.17.conv.2',
                  'encoder.features.18.0'
                  ]
    blk_id = 0
    for name, module in model.named_modules():
        if name in layer_list and isinstance(module, nn.Conv2d):
            prune_conv(module, probs[blk_id])
            blk_id += 1
    return model, enc_ch, dec_ch


class Mobilenetv2Encoder(nn.Module):
    def __init__(self, pretrained=False, progress=True):
        super(Mobilenetv2Encoder, self).__init__()
        self.num_ch_enc = np.array([32, 24, 32, 64, 1280])
        self.num_ch_dec = np.array([6, 12, 24, 32, 64])
        self.prune = prune_tp

        self.encoder = MobileNetV2()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=progress)
            self.encoder.load_state_dict(state_dict)

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
                torch.quantization.fuse_modules(m, ["0", "1"], inplace=True)
            if type(m) is InvertedResidual:
                m.fuse_model()


class ExpModel(Mobilenetv2Encoder):
    def __init__(self, pretrained=False, progress=True, mnist=False):
        super(ExpModel, self).__init__()

        self.encoder = MobileNetV2(num_classes=10, mnist=mnist)

    def forward(self, input_image):
        out = self.quant(input_image)
        out = self.encoder(out)
        out = self.dequant(out)
        return out


if __name__ == '__main__':
    net = Mobilenetv2Encoder(pretrained=True)
    print(net)
    net, enc, dec = net.prune(net, agg=True)
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
