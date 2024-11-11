
import os
import copy
import torch
import torch.nn as nn
import logging

from timm.models.layers import DropPath, trunc_normal_

from timm.models.layers import to_2tuple




class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=30, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)



class PoolAttn(nn.Module):
    """
    Implementation of pooling for PoolAttnFormer
    --pool_size: pooling size
    """

    def __init__(self, dim=30, norm_layer=GroupNorm):
        super().__init__()
        self.patch_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.patch_pool2 = nn.AdaptiveAvgPool2d((4, None))
        self.embdim_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.embdim_pool2 = nn.AdaptiveAvgPool2d((4, None))
        self.norm = norm_layer(dim)
        self.proj0 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

        self.pooling=Pooling()

    def forward(self, x):
        B, C, H, W = x.shape
        x_patch_pooling=self.pooling(x)
        x_patch_attn1 = self.patch_pool1(x)
        x_patch_attn2 = self.patch_pool2(x)
        x_patch_attn = x_patch_attn1 @ x_patch_attn2
        x_patch_attn = self.proj0(x_patch_attn)

        x1 = x.view(B, C, H * W).transpose(1, 2).view(B, H * W, C, -1)  # 64 121 30 1
        x_embdim_attn1 = self.embdim_pool1(x1)  # 64 121 30 4
        x_embdim_attn2 = self.embdim_pool2(x1)  # 64 121 4 1
        x_embdim_attn = x_embdim_attn1 @ x_embdim_attn2  # 64 121 30 1

        x_embdim_attn = x_embdim_attn.view(B, H * W, C).transpose(1, 2).view(B, C, H, W)  # 64 30 11 11
        x_embdim_attn = self.proj1(x_embdim_attn)

        x_out = self.norm(x_patch_attn+x_embdim_attn+x_patch_pooling)
        x_out = self.proj2(x_out)
        return x_out


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return x-self.pool(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.nor1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.nor2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):

    def __init__(self, dim=30, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0.2, drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # self.token_mixer = Pooling(pool_size=pool_size)
        self.token_mixer = PoolAttn(dim=dim, norm_layer=norm_layer)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolAttnFormer.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

