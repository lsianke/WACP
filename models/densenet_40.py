import torch
import torch.nn as nn
import torch.nn.functional as F

import math

norm_mean, norm_var = 0.0, 1.0

cov_cfg=[(3*i+1) for i in range(12*3+2+1)]

class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, dropRate=0):
        super(DenseBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                               padding=1, bias=False)

        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):

    def __init__(self, compress_rate, depth=40, block=DenseBasicBlock,
        dropRate=0, num_classes=10, growthRate=12, compressionRate=1):
        super(DenseNet, self).__init__()
        self.compress_rate=compress_rate

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6

        transition = Transition

        self.covcfg=cov_cfg

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = math.ceil(growthRate * 2 * (1.0 - compress_rate[0]))
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)

        self.dense1 = self._make_denseblock(block, n, compress_rate[1:n+1])
        self.trans1 = self._make_transition2(transition,self.inplanes, 168 , compress_rate[n+1])
        # self.trans1 = self._make_transition(transition, compressionRate, compress_rate[n+1])
        self.dense2 = self._make_denseblock(block, n, compress_rate[n+2:2*n+2])
        self.trans2 = self._make_transition2(transition,self.inplanes, 312 , compress_rate[2*n+2])
        # self.trans2 = self._make_transition(transition, compressionRate, compress_rate[2*n+2])
        self.dense3 = self._make_denseblock(block, n, compress_rate[2*n+3:3*n+3])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(self.inplanes, num_classes)

    def _make_denseblock(self, block, blocks, compress_rate):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, outplanes=math.ceil(self.growthRate*(1-compress_rate[i])), dropRate=self.dropRate))
            self.inplanes += math.ceil(self.growthRate*(1-compress_rate[i]))

        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate, compress_rate):
        inplanes = self.inplanes
        outplanes = int(math.ceil(self.inplanes*(1-compress_rate) // compressionRate))
        self.inplanes = outplanes
        return transition(inplanes, outplanes)

    def _make_transition2(self, transition, inplanes, outplanes, compress_rate):
        outplanes = math.ceil(outplanes * (1.0 - compress_rate)) 
        self.inplanes = outplanes
        return transition(inplanes, outplanes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def densenet_40(compress_rate = None,oristate_dict = None,ranks = None):
    model = None
    if compress_rate is None: 
        compress_rate = [0] * 40
        model = DenseNet(compress_rate=compress_rate, depth=40, block=DenseBasicBlock)
    elif oristate_dict is not None: 
        model = DenseNet(compress_rate=compress_rate, depth=40, block=DenseBasicBlock)
        # print(model)
        state_dict = model.state_dict()
        cov_id = 0
        _rank = []
        base = 0
        last_select_index = None

        for k,name in enumerate(oristate_dict):
            if k <= 233:
                if k % 6 == 0:
                    rank = ranks[cov_id]
                    if rank == 'no_pruned':
                        rank = list(range(len(state_dict[name])))
                    if len(_rank) > 0:
                        for _i,i in enumerate(rank):
                            for _j,j in enumerate(_rank):
                                state_dict[name][_i][_j] = oristate_dict[name][i][j]
                    else :
                        for _i,i in enumerate(rank):
                            state_dict[name][_i] = oristate_dict[name][i]

                    if k != 78 and k != 156:
                        _rank += [x + base for x in rank]
                        base  += len(oristate_dict[name])
                    else :
                        _rank = rank
                    cov_id += 1
                elif k % 6 == 5:
                    state_dict[name] = oristate_dict[name]
                else:
                    for _i,i in enumerate(_rank):
                        state_dict[name][_i] = oristate_dict[name][i]
            elif k == 234 :
                for _i,i in enumerate(range(10)):
                    for _j,j in enumerate(_rank):
                        state_dict[name][_i][_j] = oristate_dict[name][i][j]
            elif k == 235:
                state_dict[name] = oristate_dict[name]

        model.load_state_dict(state_dict)
    else :
        model = DenseNet(compress_rate=compress_rate, depth=40, block=DenseBasicBlock)
    return model


if __name__ == '__main__':
    model = densenet_40()
    print(model)