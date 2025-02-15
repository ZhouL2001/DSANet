import torch
import torch.nn as nn
import torch.nn.functional as F

from res2net import Res2Net50


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x)


class BackBone(nn.Module):
    def __init__(self, backbone='Res2Net50'):
        super().__init__()
        if backbone == 'Res2Net50':
            self.backbone = Res2Net50()
            self.channels = [256, 512, 1024, 2048]

    def forward(self, x):
        x_curr = []
        x_t1 = []
        x_t2 = []
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x1, x2, x3, x4 = self.backbone(x)
        x1 = x1.view(b, t, self.channels[0], x1.shape[-2], x1.shape[-1])
        x2 = x2.view(b, t, self.channels[1], x2.shape[-2], x2.shape[-1])
        x3 = x3.view(b, t, self.channels[2], x3.shape[-2], x3.shape[-1])
        x4 = x4.view(b, t, self.channels[3], x4.shape[-2], x4.shape[-1])
        for i in range(t):
            if i == 0:
                x_curr.append(x1[:, i])
                x_curr.append(x2[:, i])
                x_curr.append(x3[:, i])
                x_curr.append(x4[:, i])
            if i == 1:
                x_t1.append(x1[:, i])
                x_t1.append(x2[:, i])
                x_t1.append(x3[:, i])
                x_t1.append(x4[:, i])
            if i == 2:
                x_t2.append(x1[:, i])
                x_t2.append(x2[:, i])
                x_t2.append(x3[:, i])
                x_t2.append(x4[:, i])

        return x_curr, x_t1, x_t2


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class integrated_feature_map_fusion(nn.Module):
    def __init__(self, channels=None, emb_channel=256):
        super().__init__()
        if channels is not None:
            self.channels = channels
        else:
            self.channels = [256, 512, 1024, 2048]
        self.emb_channel = emb_channel
        self.up4 = nn.ConvTranspose2d(self.channels[3], self.channels[2], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(self.channels[3], self.channels[2])
        self.up3 = nn.ConvTranspose2d(self.channels[2], self.channels[1], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(self.channels[2], self.channels[1])
        self.up2 = nn.ConvTranspose2d(self.channels[1], self.channels[0], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(self.channels[1], self.emb_channel)

    def forward(self, x):
        dec = self.up4(x[3])
        dec = self.dec4(torch.cat([dec, x[2]], dim=1))
        dec = self.up3(dec)
        dec = self.dec3(torch.cat([dec, x[1]], dim=1))
        dec = self.up2(dec)
        dec = self.dec2(torch.cat([dec, x[0]], dim=1))

        return dec


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return out


class temporal_fusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att1 = ChannelAttention(in_channels=in_channels)
        self.channel_att2 = ChannelAttention(in_channels=in_channels)

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        batch_size, channel, height, width = x1.shape
        x1_ = x1.permute(0, 2, 3, 1).contiguous().view(-1, channel)
        x2_ = x2.permute(0, 2, 3, 1).contiguous().view(-1, channel)
        cosine_similarity = F.cosine_similarity(x1_, x2_, dim=1)
        cosine_similarity_map = cosine_similarity.view(batch_size, 1, height, width)
        similarity_att = torch.sigmoid(cosine_similarity_map)

        x1 = x1 * similarity_att
        x2 = x2 * similarity_att
        out = x1 * self.channel_att1(x2)
        out = out * self.channel_att2(out)

        return out


class self_att(nn.Module):
    def __init__(self, emb_channel):
        super().__init__()
        self.emb_channel = emb_channel
        self.q = nn.Linear(self.emb_channel, self.emb_channel)
        self.k = nn.Linear(self.emb_channel, self.emb_channel)
        self.v = nn.Linear(self.emb_channel, self.emb_channel)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = torch.sqrt(torch.tensor(self.emb_channel, dtype=torch.float32))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        dist = torch.bmm(q, k.transpose(-1, -2)) / self.scale
        dist = self.softmax(dist)

        mem = torch.bmm(dist, v)
        mem = mem.permute(0, 1, 2).view(b, c, h, w)
        return mem


class feature_bank(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.tf1 = temporal_fusion(in_channels=in_channels)
        self.tf2 = temporal_fusion(in_channels=in_channels)
        self_attention = [self_att(in_channels) for i in range(12)]
        self.self_att = nn.ModuleList(self_attention)

    def forward(self, x_curr, x_t1, x_t2):
        x_mem = self.tf1(x_t1, x_t2)
        x_feat = self.tf2(x_curr, x_mem)
        for sa in self.self_att:
            x_feat = sa(x_feat)

        return x_mem, x_feat


class stage_construction(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.f1 = nn.Sequential(
            BasicConv2d(in_planes=self.in_channels, out_planes=self.in_channels, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=self.in_channels, out_planes=self.channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.f2 = nn.Sequential(
            BasicConv2d(in_planes=self.channels[0], out_planes=self.channels[0], kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=self.channels[0], out_planes=self.channels[1], kernel_size=3, stride=2, padding=1),
        )
        self.f3 = nn.Sequential(
            BasicConv2d(in_planes=self.channels[1], out_planes=self.channels[1], kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=self.channels[1], out_planes=self.channels[2], kernel_size=3, stride=2, padding=1),
        )
        self.f4 = nn.Sequential(
            BasicConv2d(in_planes=self.channels[2], out_planes=self.channels[2], kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=self.channels[2], out_planes=self.channels[3], kernel_size=3, stride=2, padding=1),
        )

    def forward(self, feat):
        batch, channel, height, width = feat.shape
        assert channel == self.in_channels
        x1 = self.f1(feat)
        x2 = self.f2(x1)
        x3 = self.f3(x2)
        x4 = self.f4(x3)
        mem_stage = [x1, x2, x3, x4]

        return mem_stage


class channel_split_fusion(nn.Module):
    def __init__(self, emb_channel, channels):
        super().__init__()
        self.emb_channel = emb_channel
        self.channels = channels

    def forward(self, stage, mem_stage):
        assert len(stage) == len(mem_stage)
        fusion_stage_1 = []
        fusion_stage_2 = []
        for i in range(len(stage)):
            assert stage[i].shape == mem_stage[i].shape
            channel = stage[i].shape[1]
            assert channel % 2 == 0
            mid_channel = channel // 2
            stage_1 = stage[i][:, :mid_channel, :, :]
            stage_2 = stage[i][:, mid_channel:, :, :]
            mem_stage_1 = mem_stage[i][:, :mid_channel, :, :]
            mem_stage_2 = mem_stage[i][:, mid_channel:, :, :]
            fusion_stage_1.append(torch.cat([stage_1, mem_stage_1], dim=1))
            fusion_stage_2.append(torch.cat([stage_2, mem_stage_2], dim=1))

        return fusion_stage_1, fusion_stage_2


class decoder(nn.Module):
    def __init__(self, emb_channel, channels):
        super().__init__()
        self.emb_channel = emb_channel
        self.channels = channels
        self.ifmf1 = integrated_feature_map_fusion(channels=self.channels, emb_channel=self.emb_channel)
        self.ifmf2 = integrated_feature_map_fusion(channels=self.channels, emb_channel=self.emb_channel)

        self.cbr = BasicConv2d(in_planes=emb_channel*2, out_planes=emb_channel//2, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_channels=emb_channel//2, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, stage1, stage2):
        assert len(stage1) == len(stage2)
        feat1 = self.ifmf1(stage1)
        feat2 = self.ifmf2(stage2)
        pred = self.cbr(torch.cat([feat1, feat2], dim=1))
        pred = self.conv(pred)
        return pred

class my_model(nn.Module):
    def __init__(self, backbone='Res2Net50'):
        super().__init__()
        self.backbone = BackBone(backbone)
        self.emb_channel = 256
        if backbone == 'Res2Net50':
            self.channels = [256, 512, 1024, 2048]
        self.integrate_curr = integrated_feature_map_fusion(channels=self.channels, emb_channel=self.emb_channel)
        self.integrate_t1 = integrated_feature_map_fusion(channels=self.channels, emb_channel=self.emb_channel)
        self.integrate_t2 = integrated_feature_map_fusion(channels=self.channels, emb_channel=self.emb_channel)
        self.fb = feature_bank(self.emb_channel)

        self.s_c = stage_construction(in_channels=self.emb_channel, channels=self.channels)
        self.csf = channel_split_fusion(emb_channel=self.emb_channel, channels=self.channels)

        self.pre_t1 = nn.Conv2d(in_channels=self.emb_channel, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.pre_t2 = nn.Conv2d(self.emb_channel, 1, 1, 1, 0)

        self.decoder = decoder(emb_channel=self.emb_channel, channels=self.channels)

    def forward(self, x):
        x_curr, x_t1, x_t2 = self.backbone(x)
        i_curr = self.integrate_curr(x_curr)  # batch_size, channel ,H // 4, W // 4
        i_t1 = self.integrate_t1(x_t1)
        i_t2 = self.integrate_t2(x_t2)
        body_2 = self.pre_t2(i_t2)

        body_1, fb = self.fb(i_curr, i_t1, i_t2)
        body_1 = self.pre_t1(body_1)

        mem_stage = self.s_c(fb)
        stage1, stage2 = self.csf(x_curr, mem_stage)

        pred = self.decoder(stage1, stage2)

        return pred, body_1, body_2








