from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, BasicConv2d, PackSequenceWrapper

import torch.nn as nn
import torch
import torch.nn.functional as F

# Weighted Generalized Mean Pooling Layer
class WGeMHPP(nn.Module):
    def __init__(self, bin_num=[128], input_channel=128, p=6.5, eps=1.0e-6):
        super(WGeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1) * p)
        self.eps = eps
        self.weight_matrix = nn.Sequential(
            BasicConv2d(input_channel, 1, 1, 1, 0),
            nn.Softmax(-1)
        )

    def gem(self, ipts, weight):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p)*weight, (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, v]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            weight = self.weight_matrix(x)
            z = self.gem(z, weight).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)

# Fine-grained Part Sequence Learning Module
class FPSL(nn.Module):
    def __init__(self, in_channels, out_channels, halving, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                 padding=(1, 1, 1), bias=False, **kwargs):
        super(FPSL, self).__init__()
        self.halving = halving
        self.conv3d = nn.ModuleList([
            BasicConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
            for i in range(2 ** self.halving)])

    def forward(self, x):
        '''
            x: [n, c, d, h, w]
        '''
        h = x.size(3)
        split_size = int(h // 2 ** self.halving)
        feat = x.split(split_size, 3)
        feat = torch.cat([self.conv3d[i](_) for i, _ in enumerate(feat)], 3)
        feat = F.leaky_relu(feat)
        return feat

class GaitFM(BaseModel):
    """
        GaitFM: GaitFM: Fine-grained Motion Representation for Gait Recognition
    """

    def __init__(self, *args, **kargs):
        super(GaitFM, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']

        if dataset_name == 'OUMVLP':
            # For OUMVLP
            self.conv3d1 = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            self.conv3d2 = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            self.conv3d3 = nn.Sequential(
                BasicConv3d(in_c[0], in_c[1], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[1], in_c[1], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            # The LMA is followed by a spatial pooling
            # operation to reduce the feature dimensionality and further reduce the number of parameters.
            self.LMA = nn.Sequential(
                nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1)),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Spatial Pooling
            )

            self.conv3d4 = nn.Sequential(
                BasicConv3d(in_c[1], in_c[2], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[2], in_c[2], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            self.conv3d5 = nn.Sequential(
                BasicConv3d(in_c[2], in_c[3], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[3], in_c[3], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            self.conv3d3F = nn.Sequential(
                FPSL(in_c[0], in_c[1], halving=3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                FPSL(in_c[1], in_c[1], halving=3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )

            self.conv3d4F = nn.Sequential(
                FPSL(in_c[1], in_c[2], halving=3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                FPSL(in_c[2], in_c[2], halving=3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )

            self.conv3d5F = nn.Sequential(
                FPSL(in_c[2], in_c[3], halving=3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                FPSL(in_c[3], in_c[3], halving=3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )

            self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(64, in_c[-1], class_num)
            self.HPP = WGeMHPP(bin_num=[64], input_channel=in_c[-1])

        else:
            # For CASIA-B
            self.conv3d1 = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            self.conv3d2 = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            self.conv3d3 = nn.Sequential(
                BasicConv3d(in_c[0], in_c[1], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            # Local Motion Aggregation
            self.LMA = nn.MaxPool3d(
                kernel_size=(3, 1, 1), stride=(3, 1, 1))

            self.conv3d4 = nn.Sequential(
                BasicConv3d(in_c[1], in_c[2], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            self.conv3d5 = nn.Sequential(
                BasicConv3d(in_c[2], in_c[2], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )

            self.conv3d3F = FPSL(in_c[0], in_c[1], halving=3, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

            self.conv3d4F = FPSL(in_c[1], in_c[2], halving=3, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

            self.conv3d5F = FPSL(in_c[2], in_c[2], halving=3, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

            self.Head0 = SeparateFCs(128, in_c[-1], in_c[-1])
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(128, in_c[-1], class_num)
            self.HPP = WGeMHPP(bin_num=[128], input_channel=in_c[-1])

        self.FP = PackSequenceWrapper(torch.max)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, d, h, w = sils.size()
        if d < 3:
            repeat = 3 if d == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1) # [n, c, d, h, w]
        # shallow feature extraction
        outs = self.conv3d1(sils) # [n, c, d, h, w]
        outs = self.conv3d2(outs) # [n, c, d, h, w]
        fouts = outs
        # global branch
        outs = self.conv3d3(outs) # [n, c, d, h, w]
        outs = self.LMA(outs)
        outs = self.conv3d4(outs) # [n, c, d, h, w]
        outs = self.conv3d5(outs) # [n, c, d, h, w]
        # fine-grained branch
        fouts = self.conv3d3F(fouts) # [n, c, d, h, w]
        fouts = self.LMA(fouts)
        fouts = self.conv3d4F(fouts) # [n, c, d, h, w]
        fouts = self.conv3d5F(fouts) # [n, c, d, h, w]

        outs = torch.cat([outs, fouts], 3)

        # frame pooling
        outs = self.FP(outs, dim=2, seq_dim=2, seqL=seqL)[0] # [n, c, h, w]
        print(outs.size())

        outs = self.HPP(outs)
        outs = outs.permute(2, 0, 1).contiguous()  # [v, n, c]

        gait = self.Head0(outs)  # [v, n, c]
        gait = gait.permute(1, 2, 0).contiguous()  # [n, c, v]

        bnft = self.Bn(gait)  # [n, c, v]
        logi = self.Head1(bnft.permute(2, 0, 1).contiguous())  # [v, n, c]
        print(logi.size())

        gait = gait.permute(0, 2, 1).contiguous()  # [n, v, c]
        bnft = bnft.permute(0, 2, 1).contiguous()  # [n, v, c]
        logi = logi.permute(1, 0, 2).contiguous()  # [n, v, c]

        n, _, d, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': bnft, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*d, 1, h, w)
            },
            'inference_feat': {
                'embeddings': bnft
            }
        }
        return retval
