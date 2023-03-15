import torch.nn as nn
import math
import pdb


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,hidden_dim=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if hidden_dim is None : 
            hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, compress_rate, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t-ex, c-channel, n-blocknum, s-stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.interverted_residual_setting = interverted_residual_setting
        # if compress_rate is None: compress_rate = [0.]*20
        self.compress_rate=compress_rate[:]

        # building first layer
        assert input_size % 32 == 0 
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        _cnt = 1
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            # _output_channel = output_channel
            output_channel = math.ceil((1-self.compress_rate[_cnt])*output_channel)
            # print(_cnt,n)
            # print('1--->',self.compress_rate[_cnt])
            # _input_channel = input_channel
            _cnt += 1
            for i in range(n):
                # print('2--->',self.compress_rate[_cnt])
                if t != 1:
                    hidden_dim = math.ceil(int(input_channel * t)*(1-self.compress_rate[_cnt]))
                else :
                    hidden_dim = None
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t,hidden_dim=hidden_dim))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t,hidden_dim=hidden_dim))
                input_channel = output_channel
                _cnt += 1

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        #self.classifier = nn.Linear(self.last_channel, n_class)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(compress_rate=None,oristate_dict = None,ranks = None):
    model = None 
    if oristate_dict is not None and compress_rate is not None:
        model = MobileNetV2(compress_rate=compress_rate)
        state_dict = model.state_dict()
        cov_id = 0
        rank = None
        last_select_index = None

        # print(model)

        for k,name in enumerate(oristate_dict):
            if k < 52 * 6:
                if k%6 == 0:
                    rank = ranks[cov_id]
                    f, c, w, h = state_dict[name].size()
                    if rank == 'no_pruned':
                        rank = list(range(len(state_dict[name])))
                    # print(k,name,len(rank))
                    if last_select_index is not None:
                        if c == 1:
                            for _i,i in enumerate(rank):
                                state_dict[name][_i] = oristate_dict[name][i]
                        else :
                            for _i,i in enumerate(rank):
                                for _j,j in enumerate(last_select_index):
                                    state_dict[name][_i][_j] = oristate_dict[name][i][j]
                    else :
                        for _i,i in enumerate(rank):
                            state_dict[name][_i] = oristate_dict[name][i]
                    last_select_index = rank
                    cov_id += 1
                elif k%6 == 5:
                    state_dict[name] = oristate_dict[name]
                else:
                    for _i,i in enumerate(rank):
                        state_dict[name][_i] = oristate_dict[name][i]
            elif k == 52 * 6:
                rank = list(range(1000))
                for _i,i in enumerate(rank):
                    for _j,j in enumerate(last_select_index):
                        state_dict[name][_i][_j] = oristate_dict[name][i][j]
            elif k == 52 * 6 + 1:
                state_dict[name] = oristate_dict[name]

        model.load_state_dict(state_dict)

    elif compress_rate is not None:
        model = MobileNetV2(compress_rate=compress_rate)
    else :
        compress_rate = [0.]*100
        model = MobileNetV2(compress_rate=compress_rate)
    return model

if __name__ == '__main__':
    model = mobilenet_v2()
    print(model)
