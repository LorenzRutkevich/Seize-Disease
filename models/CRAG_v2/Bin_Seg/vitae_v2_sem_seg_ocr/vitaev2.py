from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
import torch.utils.checkpoint as checkpoint
from .NormalCell import NormalCell
from .ReductionCell import ReductionCell
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
from mmcv.cnn import ConvModule
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48, img_size=None):
        self.img_size = img_size
        self.inter_channel = inter_channel
        self.out_channel = out_channels
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size):
        x = self.conv3(self.conv2(self.conv1(x)))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        return x, (h, w)

    def flops(self, ) -> float:
        flops = 0
        flops += 3 * self.inter_channel * self.img_size[0] * self.img_size[1] // 4 * 9
        flops += self.img_size[0] * self.img_size[1] // 4 * self.inter_channel
        flops += self.inter_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16 * 9
        flops += self.img_size[0] * self.img_size[1] // 16 * self.out_channel
        flops += self.out_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16
        return flops

class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7, RC_heads=1, NC_heads=6, dilations=[1, 2, 3, 4],
                RC_op='cat', RC_tokens_type='performer', NC_tokens_type='transformer', RC_group=1, NC_group=64, NC_depth=2, dpr=0.1, mlp_ratio=4., qkv_bias=True, 
                qk_scale=None, drop=0, attn_drop=0., norm_layer=nn.LayerNorm, class_token=False, gamma=False, init_values=1e-4, SE=False, window_size=7,
                use_checkpoint=False):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = self.img_size // self.downsample_ratios
        self.RC_kernel_size = kernel_size
        self.RC_heads = RC_heads
        self.NC_heads = NC_heads
        self.dilations = dilations
        self.RC_op = RC_op
        self.RC_tokens_type = RC_tokens_type
        self.RC_group = RC_group
        self.NC_group = NC_group
        self.NC_depth = NC_depth
        self.use_checkpoint = use_checkpoint
        if RC_tokens_type == 'stem':
            self.RC = PatchEmbedding(inter_channel=token_dims//2, out_channels=token_dims, img_size=img_size)
        elif downsample_ratios > 1:
            self.RC = ReductionCell(img_size, in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
                            RC_heads, dilations, op=RC_op, tokens_type=RC_tokens_type, group=RC_group, gamma=gamma, init_values=init_values, SE=SE)
        else:
            self.RC = nn.Identity()
        self.NC = nn.ModuleList([
            NormalCell(token_dims, NC_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                       drop_path=dpr[i] if isinstance(dpr, list) else dpr, norm_layer=norm_layer, class_token=class_token, group=NC_group, tokens_type=NC_tokens_type,
                       gamma=gamma, init_values=init_values, SE=SE, img_size=img_size // downsample_ratios, window_size=window_size, shift_size=0)
        for i in range(NC_depth)])

    def forward(self, x, size):
        h, w = size
        x, (h, w) = self.RC(x, (h, w))
        # print(h, w)
        for nc in self.NC:
            nc.H = h
            nc.W = w
            if self.use_checkpoint:
                x = checkpoint.checkpoint(nc, x)
            else:
                x = nc(x)
            # print(h, w)
        return x, (h, w)

class ViTAEv2(nn.Module):
    def __init__(self,
                img_size=256,
                in_chans=3,
                embed_dims=32,
                token_dims=32,
                downsample_ratios=[4, 2, 2, 2],
                kernel_size=[8, 3, 3, 3],
                RC_heads=[1, 1, 1, 1],
                NC_heads=4,
                dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
                RC_op='cat',
                RC_tokens_type='window',
                NC_tokens_type='transformer',
                RC_group=[1, 1, 1, 1],
                NC_group=[1, 16, 32, 32],
                NC_depth=[2, 2, 6, 2], 
                mlp_ratio=4.,
                qkv_bias=True, 
                qk_scale=None, 
                drop_rate=0.1, 
                attn_drop_rate=0., 
                drop_path_rate=0., 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                stages=4,
                window_size=8,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                use_checkpoint=False,
                load_ema=True,
                use_seg_head=True,):
        super().__init__()

        self.stages = stages
        self.load_ema = load_ema
        repeatOrNot = (lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)])
        self.embed_dims = repeatOrNot(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in range(stages)]
        self.downsample_ratios = repeatOrNot(downsample_ratios, stages)
        self.kernel_size = repeatOrNot(kernel_size, stages)
        self.RC_heads = repeatOrNot(RC_heads, stages)
        self.NC_heads = repeatOrNot(NC_heads, stages)
        self.dilaions = repeatOrNot(dilations, stages)
        self.RC_op = repeatOrNot(RC_op, stages)
        self.RC_tokens_type = repeatOrNot(RC_tokens_type, stages)
        self.NC_tokens_type = repeatOrNot(NC_tokens_type, stages)
        self.RC_group = repeatOrNot(RC_group, stages)
        self.NC_group = repeatOrNot(NC_group, stages)
        self.NC_depth = repeatOrNot(NC_depth, stages)
        self.mlp_ratio = repeatOrNot(mlp_ratio, stages)
        self.qkv_bias = repeatOrNot(qkv_bias, stages)
        self.qk_scale = repeatOrNot(qk_scale, stages)
        self.drop = repeatOrNot(drop_rate, stages)
        self.attn_drop = repeatOrNot(attn_drop_rate, stages)
        self.norm_layer = repeatOrNot(norm_layer, stages)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.with_seg = use_seg_head

        depth = np.sum(self.NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i==0 else self.NC_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i], self.NC_depth[i], dpr[startDpr:self.NC_depth[i]+startDpr],
                mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i], drop=self.drop[i], attn_drop=self.attn_drop[i],
                norm_layer=self.norm_layer[i], window_size=window_size, use_checkpoint=use_checkpoint)
            )
            img_size = img_size // self.downsample_ratios[i]
            in_chans = self.tokens_dims[i]

        self.layers = nn.ModuleList(Layers)
        self.num_layers = len(Layers)

        self._freeze_stages()

    def _freeze_stages(self):

        if self.frozen_stages > 0:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def forwardTwoLayer(self, x):
        x1 = self.layers[0](x)
        x2 = self.layers[1](x1)
        return x1, x2
    
    def forwardThreeLayer(self, x):
        x0 = self.layers[1](x)
        x1 = self.layers[2](x0)
        x2 = self.layers[3](x1)
        return x0, x1, x2

    def forward_features(self, x):
        b, c, h, w = x.shape
        for layer in self.layers:
            x, (h, w) = layer(x, (h, w))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def forward(self, x):
        """Forward function."""

        outs = []

        b, _, h, w = x.shape
        for layer in self.layers:
            x, (h, w) = layer(x, (h, w))
            outs.append(x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous())

        outs = [outs[i] for i in self.out_indices]


        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(ViTAEv2, self).train(mode)
        self._freeze_stages()

class SpatialGather(Module):

    def __init__(self, num_classes: int, scale: float = 1):
        super(SpatialGather, self).__init__()
        self.cls_num = num_classes
        self.scale = scale

    def forward(self, feats, probs):
        probs = rearrange(probs, 'b k h w -> b k (h w)')  # b x k x (h w)
        feats = rearrange(feats, 'b c h w -> b (h w) c')  # b x (h w) x c
        probs = torch.softmax(self.scale * probs, dim=-1)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = rearrange(ocr_context, 'b k c -> b c k 1')
        return ocr_context


class ObjectAttentionBlock2D(Module):

    def __init__(self, in_channels: int, key_channels: int, scale: int = 1, bn_type='BN'):
        super(ObjectAttentionBlock2D, self).__init__()
        self.scale = scale
        self.key_channels = key_channels
        self.pool = MaxPool2d(kernel_size=(scale, scale)) if scale > 1 else Identity()
        self.up = UpsamplingBilinear2d(scale_factor=scale) if scale > 1 else Identity()
        self.to_q = Sequential(
            ConvModule(in_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type)),
            ConvModule(key_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type))
        )
        self.to_k = Sequential(
            ConvModule(in_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type)),
            ConvModule(key_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type))
        )
        self.to_v = ConvModule(in_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type))
        self.f_up = ConvModule(key_channels, in_channels, 1, bias=False, norm_cfg=dict(type=bn_type))

    def forward(self, feats, context):
        b, c, h, w = feats.shape
        feats = self.pool(feats)

        query = rearrange(self.to_q(feats), 'b c h w -> b (h w) c ')
        key = rearrange(self.to_k(context), 'b c k 1 -> b c k')
        value = rearrange(self.to_v(context), 'b c k 1 -> b k c ')

        sim_map = torch.matmul(query, key)  # b l k
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)  # b l c

        context = rearrange(context, 'b (h w) c ->b c h w', h=h, w=w)

        context = self.f_up(context)
        context = self.up(context)

        return context


class SpatialOCR(Module):

    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1, bn_type='BN'):
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale, bn_type)
        self.conv_bn_dropout = Sequential(
            ConvModule(2 * in_channels, out_channels, 1, bias=False, norm_cfg=dict(type='BN')),
            Dropout2d(dropout)
        )

    def forward(self, feats, context):
        context = self.object_context_block(feats, context)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class OCRNet(Module):
    def __init__(self, channels, num_classes, ocr_mid_channels=512, ocr_key_channels=256):
        super(OCRNet, self).__init__()

        self.soft_object_regions = Sequential(
            ConvModule(channels, channels, 1),
            Conv2d(channels, num_classes, 1)
        )

        self.pixel_representations = ConvModule(channels, ocr_mid_channels, 3, 1, 1)

        self.object_region_representations = SpatialGather(num_classes)

        self.object_contextual_representations = SpatialOCR(in_channels=ocr_mid_channels,
                                                            key_channels=ocr_key_channels,
                                                            out_channels=ocr_mid_channels,
                                                            scale=1,
                                                            dropout=0.05,
                                                            )
        self.augmented_representation = Conv2d(ocr_mid_channels, num_classes, kernel_size=1)

    def forward(self, feats):
        out_aux = self.soft_object_regions(feats)  # b k h w

        feats = self.pixel_representations(feats)  # b c h w

        context = self.object_region_representations(feats, out_aux)  # b c k 1

        feats = self.object_contextual_representations(feats, context)  # b c h w

        out = self.augmented_representation(feats)  # b k h w

        return out_aux, out
    
class FinalBlockInter(nn.Module):
    def __init__(self, in_channels, task='binary', dropout_ratio=0.):
        super(FinalBlockInter, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.mish1 = nn.Mish(inplace=True)
        self.dropout = nn.Dropout2d(p=self.dropout_ratio)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.mish2 = nn.Mish(inplace=True)
        self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=1)
        if task == 'binary':
            self.act = nn.Sigmoid()
        elif task == 'multiclass':
            self.act = nn.Softmax(dim=1)


    def forward(self, x, input_size):
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mish1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mish2(x)
        x = self.conv3(x)
        x = self.act(x)
        return x

class FinalBlockUp(nn.Module):
    def __init__(self, task='binary', fast=False, dropout_ratio=0., c=64, h=64, w=64):
        super(FinalBlockUp, self).__init__()
        self.task = task
        self.fast = fast
        self.dropout_ratio = dropout_ratio
        self.c = c
        self.h = h
        self.w = w

        self.upsample1 = nn.Upsample(size=(h * 2, w * 2), mode='bilinear', align_corners=False)
        self.batchnorm1 = nn.BatchNorm2d(c)
        self.mish1 = nn.Mish(inplace=True)
        self.dropout1 = nn.Dropout2d(p=self.dropout_ratio)
        self.conv1 = nn.Conv2d(c, c // 2, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(c // 2 if c > 1 else c)
        self.mish2 = nn.Mish(inplace=True)
        self.upsample2 = nn.Upsample(size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        self.batchnorm3 = nn.BatchNorm2d(c // 2 if c > 1 else c)
        self.mish3 = nn.Mish(inplace=True)
        self.dropout2 = nn.Dropout2d(p=self.dropout_ratio)
        self.conv2 = nn.Conv2d(c // 2, c, kernel_size=1)
        self.activation = nn.Sigmoid() if self.task == 'binary' else nn.Softmax(dim=1)
        self.f_upsample1 = nn.Upsample(size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        self.f_conv1 = nn.Conv2d(c, 1, kernel_size=c)

    def forward(self, x):
        b, c, h, w = x.shape
        if not self.fast:
            x = self.upsample1(x)
           # print(x.shape, self.h * 2, self.w * 2)
            x = self.batchnorm1(x)
            x = self.mish1(x)
            x = self.dropout1(x)
            if c >= 2:
                x = self.conv1(x)
                x = self.batchnorm2(x)
                x = self.mish2(x)
            x = self.upsample2(x)
           # print(x.shape, self.h * 4, self.w * 4)
            x = self.batchnorm3(x)
            if c // 2 >= 1:
                x = self.mish3(x)
                x = self.dropout2(x)
                x = self.conv2(x)
                x = self.activation(x)
            elif c == 1:
                x = self.activation(x)
        else:
            #print("fast")
            x = self.f_upsample1(x)
            x = self.batchnorm1(x)
            x = self.mish1(x)
            x = self.dropout1(x)
            x = self.f_conv1(x)
            x = self.activation(x)
        return x

    
class ViTAEv2_OCR(nn.Module):
    def __init__(self, img_size: int = 256, num_classes: int = 1, in_channels: int = 3, task: str = 'binary', fast: bool = False, inter: bool = False, dropout_ratio: float = 0.,
                vitae_args: dict = None, ocr_args: dict = None, inter_args: dict = None, up_args: dict = None):
        super(ViTAEv2_OCR, self).__init__()
        self.inter = inter
        if task in ['binary', 'multiclass']:
            self.task = task
        else:
            raise ValueError('Task must be binary or multiclass')
        self.fast = fast
        self.img_size = img_size
        self.num_classes = num_classes
        self.vitae_args = vitae_args
        self.ocr_args = ocr_args
        self.dropout_ratio = dropout_ratio
        self.inter_args = inter_args
        self.up_args = up_args
        self.in_channels = in_channels

        self.vita = ViTAEv2(img_size=self.img_size, in_chans=self.in_channels) if self.vitae_args is None else ViTAEv2(**self.vitae_args)
        self.ocr = OCRNet(channels=32, num_classes=self.num_classes) if self.ocr_args is None else OCRNet(**self.ocr_args)
        if self.inter:
            self.head = FinalBlockInter(in_channels=self.num_classes, task=self.task, dropout_ratio=self.dropout_ratio) if self.inter_args is None else FinalBlockInter(**self.inter_args)
        else:
            self.head2 = FinalBlockUp(task=self.task, fast=self.fast, dropout_ratio=self.dropout_ratio, c=num_classes, h=img_size//4, w=img_size//4) if self.up_args is None else FinalBlockUp(**self.up_args)
    def forward(self, x):
        out = self.vita(x)
        out_aux, out = self.ocr(out[0])
        if self.inter:
            out = self.head(out, input_size=self.img_size)
        else:
            out = self.head2(out)
        return out

vitae_args_test = {
    'img_size': 128,
    'in_chans': 1,
    'embed_dims': 32,
    'token_dims': 32,
    'downsample_ratios': [4, 2, 2, 2],
    'kernel_size': [8, 3, 3, 3],
    'RC_heads': [1, 1, 1, 1],
    'NC_heads': 4,
    'dilations': [[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
    'RC_op': 'cat',
    'RC_tokens_type': 'window',
    'NC_tokens_type': 'transformer',}

ocr_args_test = {
    'channels': 32,
    'num_classes': 1,
    'ocr_mid_channels': 512,
    'ocr_key_channels': 256,}

inter_args_test = {
    'in_channels': 32,
    'num_classes': 1,
    'task': 'binary',
    'dropout_ratio': 0.1,}

up_args_test = {
    'task': 'binary',
    'fast': False,
    'dropout_ratio': 0.1,}


# test code

#if __name__ == '__main__':
    #vita = ViTAEv2(img_size=256, in_chans=1)
    #dummy = torch.randn(2, 1, 256, 256)
    #out = vita(dummy)
    # print(out[0].shape)
    #print(out[1].shape)
    #ocr = OCRNet(channels=out[0].shape[1], num_classes=1)
    #out_aux, out = ocr(out[0])
    #print(out.shape)
    #head2 = FinalBlockUp()
    #out = head2(out, fast=True)
    #print(out.shape)
    #dummy = torch.randn(2, 3, 256, 256)
    #model = ViTAEv2_OCR(img_size=256, num_classes=1, task='binary', fast=True,inter=True, in_channels=3)
    #out = model(dummy)
    #print(out.shape)
    #from thop import profile
    #flops, params = profile(model, inputs=(dummy, ))
    #print(flops, params)
    #print(torch.min(out), torch.max(out))
