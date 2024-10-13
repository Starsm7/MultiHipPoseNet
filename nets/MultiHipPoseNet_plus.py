import torch
from fvcore.nn import parameter_count_table
from thop import profile
import math
import torch.nn as nn
from torchviz import make_dot
from nets.modules.config import get_parser
from nets.modules.model_utils import catUp, AdaptiveFeatureFusionModule, Bottleneck
import torch.nn.functional as F

# model file


# Expert network definition, you can choose any network you want
class Expert(nn.Module):
    def __init__(self, in_channels,block=Bottleneck, layers=[3, 4, 6, 3]):
        super(Expert, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)
        x = self.maxpool(feat1)
        feat2 = self.layer1(x)
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]



# ME-GCT
class ExpertGate(nn.Module):
    def __init__(self, in_channels, n_expert, n_task, use_gate=False):
        super(ExpertGate, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate
        self.n_expert = n_expert
        # Creating multiple expert networks
        self.expert_layers = nn.ModuleList([Expert(in_channels) for _ in range(n_expert)])
        # Fusion modules for feature combination
        self.fusion_modules = nn.ModuleList(
            [AdaptiveFeatureFusionModule(channels, n_expert) for channels in [64, 256, 512, 1024, 2048]])

    def forward(self, x):
        # Forward pass through all expert networks
        expert_outputs = [expert(x) for expert in self.expert_layers]
        towers = []
        if self.use_gate:
            # Using ME-GCT gating mechanism for task-specific outputs
            for task_index in range(self.n_task):
                tower = []
                for index, fusion_module in enumerate(self.fusion_modules):
                    e_net=[]
                    for i in range(self.n_expert):
                        e_net.append(expert_outputs[i][index])
                    out = fusion_module(*e_net)
                    tower.append(out)
                towers.append(tower)
        else:
            # Averaging expert outputs for each layer
            for index, _ in enumerate(self.fusion_modules):
                e_net = []
                for i in range(self.n_expert):
                    e_net.append(expert_outputs[i][index])
                out = sum(e_net) / len(e_net)
                towers.append(out)
        return towers

# Multi-Task Hip Joint Structure and Key Point Prediction Model
class MultiHipPoseNet(nn.Module):
    '''
    hip_classes:number of key anatomical structures
    kpt_n:number of key points
    n_expert:number of experts
    n_task:number of tasks, structure segmentation and keypoint detection
    in_channels:number of channels in the image
    use_gate:whether to use ME-GCT or not
    '''
    def __init__(self, hip_classes, kpt_n, n_expert, n_task, in_channels=3, layer = 4, use_gate=False):
        super(MultiHipPoseNet, self).__init__()
        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]
        self.n_task = n_task
        self.layer = layer

        # Concatenation layers for upsampling
        self.up_concat4 = nn.ModuleList([catUp(in_filters[3], out_filters[3]) for _ in range(self.n_task)])
        self.up_concat3 = nn.ModuleList([catUp(in_filters[2], out_filters[2]) for _ in range(self.n_task)])
        self.up_concat2 = nn.ModuleList([catUp(in_filters[1], out_filters[1]) for _ in range(self.n_task)])
        self.up_concat1 = nn.ModuleList([catUp(in_filters[0], out_filters[0]) for _ in range(self.n_task)])

        # Upsampling and convolution layers
        self.up_conv = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)) for _ in range(self.n_task)])
        
        # ME-GCT
        self.use_gate = use_gate
        self.Expert_Gate = ExpertGate(in_channels=in_channels, n_expert=n_expert, n_task=n_task, use_gate=use_gate)
        self.adp = nn.ModuleList([GraphGenerator(out_filters[i]) for i in range(self.layer)])

        # Final output layers for each task
        self.final = nn.ModuleList([nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hip_classes, 3, padding=1)),
            nn.Sequential(
                nn.GroupNorm(32, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, kpt_n, 3, padding=1))])

    def forward(self, x):
        towers = self.Expert_Gate(x)
        final = []


        if self.use_gate:
            # calculate the current adaptive adj matrix once per iteration
            adp = [self.adp[i]() for i in range(self.layer)]
            Up1, Up2, Up3, Up4 = [], [], [], []
            # Using ME-GCT gating mechanism for upsampling
            for index in range(self.n_task):
                up4 = self.up_concat4[index](towers[index][3], towers[index][4])
                up3 = self.up_concat3[index](towers[index][2], up4)
                up2 = self.up_concat2[index](towers[index][1], up3)
                up1 = self.up_concat1[index](towers[index][0], up2)
                Up1.append(up1), Up2.append(up2), Up3.append(up3), Up4.append(up4)

            for index in range(self.n_task):
                index_adp = 0 if index == 1 else 1
                up4 = Up4[index] + torch.einsum('bchw,cd->bdhw', Up4[index_adp], adp[3])
                up3 = self.up_concat3[index](towers[index][2], up4) + torch.einsum('bchw,cd->bdhw', Up3[index_adp], adp[2])
                up2 = self.up_concat2[index](towers[index][1], up3) + torch.einsum('bchw,cd->bdhw', Up2[index_adp], adp[1])
                up1 = self.up_concat1[index](towers[index][0], up2) + torch.einsum('bchw,cd->bdhw', Up1[index_adp], adp[0])
                up1 = self.up_conv[index](up1)
                final_output = self.final[index](up1)
                final.append(final_output)
        else:
            # Upsampling without ME-GCT gating mechanism
            for index in range(self.n_task):
                up4 = self.up_concat4[index](towers[3], towers[4])
                up3 = self.up_concat3[index](towers[2], up4)
                up2 = self.up_concat2[index](towers[1], up3)
                up1 = self.up_concat1[index](towers[0], up2)
                up1 = self.up_conv[index](up1)
                final_output = self.final[index](up1)
                final.append(final_output)
        return final
    
    # Generate a summary of the network
    def summary(self, net):
        x = torch.rand(4, 3, get_parser().input_h, get_parser().input_w).to('cuda')
        x1, x2 = net(x)
        print(parameter_count_table(net))
        print(x1.shape, x2.shape)
        flops, params = profile(m, inputs=(x,))
        print(f"FLOPs: {flops / 1e9 :.2f} GFLOPs")
        print(f"Params: {params / 1e6 :.2f} M")
        dot = make_dot((x1, x2), params=dict(m.named_parameters()))
        dot.render('./MultiHipPoseNet', format='pdf')

def downsample_to_target_size(matrix, target_size):
    original_shape = matrix.shape
    assert original_shape[0] == original_shape[1], "Matrix must be square"
    shrink_factor = original_shape[0] // target_size
    max_size = shrink_factor * target_size
    trimmed_matrix = matrix[:max_size, :max_size]
    downsampled_matrix = trimmed_matrix.reshape(target_size, shrink_factor, target_size, shrink_factor).sum(axis=(1, 3))
    return downsampled_matrix

class GraphGenerator(nn.Module):
    def __init__(self, channels):
        super(GraphGenerator, self).__init__()
        self.channels = channels
        self.nodevec1 = nn.Parameter(torch.randn(1024, 10).to('cuda'), requires_grad=True).to('cuda')
        self.nodevec2 = nn.Parameter(torch.randn(10, 1024).to('cuda'), requires_grad=True).to('cuda')
    def forward(self):
        adp = F.softmax(downsample_to_target_size(F.relu(torch.mm(self.nodevec1, self.nodevec2)), self.channels), dim=1)
        return adp

if __name__ == "__main__":
    m = MultiHipPoseNet(hip_classes=8, kpt_n=6, n_expert=3, n_task=2, use_gate=True).to('cuda')
    m.summary(m)
