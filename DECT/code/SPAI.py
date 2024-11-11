import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
# from basicsr.utils.registry import ARCH_REGISTRY
from PAT import PoolFormerBlock


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos



class SPAI(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.3, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        # self.model = MFF(dim=dim, ffn_expansion_factor=1, bias=0, act=False)
        self.model = PoolFormerBlock(dim=dim)

    def forward(self, x, use_DropKey=True):

        H = x.shape[2]
        W = x.shape[3]
        x = x.permute(0, 2, 3, 1)
        x = rearrange(x, 'b h w c ->b (h w) c')
        # ---------------------------SA------------------------------
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        v_1 = q.reshape(B, C, N).contiguous().view(B, C, H, W)
        v_2 = k.reshape(B, C, N).contiguous().view(B, C, H, W)
        v_3 = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # use DropKey as regularizer
        attn = self.dropkey(attn)
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        #attened_x = x#NO SA
        # ------------------------------------------------------------------

        # PAT
        conv_x = self.model(v_3)
        # conv_x = x.transpose(-2, -1).contiguous().view(B, C, H, W)# NO PAT

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, 1)

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)

        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        # conv_x2 = conv_x2.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        # x=torch.unsqueeze(x,dim=1)

        return x

    def dropkey(selsf, attn):
        m_r = torch.ones_like(attn) * 0.5
        attn = attn + torch.bernoulli(m_r) * -1e-12
        return attn


class MFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, act=True):
        super(MFF, self).__init__()
        self.act = act
        hidden_features = int(dim * ffn_expansion_factor)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=hidden_features, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
        )

    def forward(self, x):

        sca = self.sca(x)
        xd = self.dwconv(x)
        xb = self.bc(x)

        xdb = F.gelu(xd) * xb if self.act else xd * xb
        x1 = xdb * sca
        x2 = self.project_out(x1)
        x = x + x2
        return x

    def bc(self, x):
        M1 = torch.sigmoid(x)
        M2 = F.elu(x)
        reward = 2 * M1
        punishment = torch.zeros_like(M1)
        M1 = torch.where(M1 > 0.2, reward, punishment)
        A = torch.mul(M1, M2)

        return A

    def LN(self, x):
        LayerNorm = nn.LayerNorm(x.size()[1:])
        x = LayerNorm(x)
        return x

