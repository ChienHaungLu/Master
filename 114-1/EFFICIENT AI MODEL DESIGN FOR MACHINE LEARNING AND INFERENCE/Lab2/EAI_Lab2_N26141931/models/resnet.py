
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Bottleneck Block
# ----------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        assert len(out_channels) == 3, "Bottleneck requires out_channels list of length 3"
        
        ################################################
        # Please replace ??? with the correct variable #
        # example: in_channels, out_channels[0], ...   #
        ################################################
        # out_channels: [C1, C2, C3] correspond to conv1, conv2, conv3 *intermediate* widths
        # IMPORTANT: conv3/bn3 is kept FIXED to planes * expansion (bn3 not changed/pruned).
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        # Keep bn3 (and conv3 output width) unchanged: planes * expansion
        c3 = planes * self.expansion
        self.conv3 = nn.Conv2d(out_channels[1], c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out

# ----------------------
# ResNet
# ----------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, cfg, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        # cfg should be: [64] + [c1,c2,c3]*sum(layers); we don't prune bn3 so c3 entries are ignored for conv3 width
        assert cfg is not None and len(cfg) == (1 + 3*sum(layers)), "cfg length mismatch"
        self.current_cfg_idx = 0

        # Conv1: CIFAR-10 setting (3x3, stride 1), fixed to 64 out channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Skip cfg[0] which corresponds to conv1's out channels (64)
        self.current_cfg_idx += 1
        self.inplanes = 64

        # Layer1~Layer4
        self.layer1 = self._make_layer(block, 64, layers[0], cfg, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], cfg, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        #############################################################################
        # Generate the correct layers and downsample based on cfg, while keeping    #
        # bn3 (and thus bottleneck output width) FIXED to planes*expansion.         #
        #############################################################################
        layers = []
        c3 = planes * block.expansion  # fixed output width for this stage

        for i in range(blocks):
            # Take three entries from cfg for this bottleneck: conv1, conv2, conv3 (conv3 is unused for width)
            out_ch = cfg[self.current_cfg_idx : self.current_cfg_idx + 3]
            if len(out_ch) != 3:
                raise ValueError(f"cfg length mismatch when constructing bottleneck: current_idx {self.current_cfg_idx}, cfg_len {len(cfg)}")
            self.current_cfg_idx += 3

            if i == 0:
                # first block in this layer: may need downsample if stride != 1 or channel mismatch
                downsample = None
                if stride != 1 or self.inplanes != c3:
                    # Projection shortcut: 1x1 conv (NO BN) as per lab to ease pruning copying
                    downsample = nn.Conv2d(self.inplanes, c3, kernel_size=1, stride=stride, bias=False)
                layers.append(block(self.inplanes, planes, out_ch, downsample=downsample, stride=stride))
                self.inplanes = c3
            else:
                # identity shortcut, stride = 1
                layers.append(block(self.inplanes, planes, out_ch, downsample=None, stride=1))
                self.inplanes = c3

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*6 + \
              [512, 512, 2048]*3
    layers = [3, 4, 6, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)

def ResNet101(num_classes=10, in_channels=3, cfg=None):
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*23 + \
              [512, 512, 2048]*3
    layers = [3, 4, 23, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)

def ResNet152(num_classes=10, in_channels=3, cfg=None):
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*8 + \
              [256, 256, 1024]*36 + \
              [512, 512, 2048]*3
    layers = [3, 8, 36, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)
