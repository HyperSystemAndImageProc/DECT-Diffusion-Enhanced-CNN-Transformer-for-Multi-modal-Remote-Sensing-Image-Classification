import torch
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange, repeat

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class BottleStack(nn.Module):
    def __init__(
            self,
            *,
            dim=64,
            fmap_size,
            dim_out=64,
            proj_factor=6,
            num_layers=3,
            heads=2,
            dim_head=10,
            downsample=True,
            rel_pos_emb=False,
            activation=nn.ReLU()
    ):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample
            layer_fmap_size = fmap_size // (2 if downsample and not is_first else 1)

            layers.append(BottleBlock(
                dim=dim,
                fmap_size=layer_fmap_size,
                dim_out=dim_out,
                proj_factor=proj_factor,
                heads=heads,
                dim_head=dim_head,
                downsample=layer_downsample,
                rel_pos_emb=rel_pos_emb,
                activation=activation
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(dim=4)
        _, c, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert h == self.fmap_size and w == self.fmap_size, f'height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}'
        result = self.net(x)
        return torch.unsqueeze(result, dim=4)


class BottleBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            fmap_size,
            dim_out,
            proj_factor,
            downsample,
            heads=4,
            dim_head=128,
            rel_pos_emb=False,
            activation=nn.ReLU()
    ):
        super().__init__()

        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion

        attention_dim = dim_out // proj_factor

        self.net = nn.Sequential(
            nn.Conv2d(dim, attention_dim, 1, bias=False),
            nn.BatchNorm2d(attention_dim),
            activation,
            D2SA(
                dim=attention_dim,
                fmap_size=fmap_size,
                heads=heads,
                dim_head=dim_head,
                rel_pos_emb=rel_pos_emb),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(2 * attention_dim),
            activation,
            nn.Conv2d(2 * attention_dim, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out)

        )

        # init last batch norm gamma to zero

        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)


class D2SA(nn.Module):
    def __init__(
            self,
            *,
            dim,
            fmap_size,
            heads=4,
            dim_head=128,
            rel_pos_emb=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.temperature = nn.Parameter(torch.ones(dim_head))
        self.temperature1 = nn.Parameter(torch.ones(1))
        inner_dim = heads * dim_head
        norm_layer = nn.BatchNorm2d
        midplanes = outplanes = inplanes = 64
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self._up_kwargs = up_kwargs
        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, 1, bias=False)
        self.SPFC = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(128, 96),
            # mish(),
            nn.Linear(96, 64),
            # nn.Linear(64, 20)
            # nn.Softmax(dim = 1),
        )
        self.spconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, padding=(0, 0),
                      kernel_size=(1, 2), stride=(1, 1)),
            mish()
        )
        self.SPFC = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            # mish(),
            nn.Linear(32, 64),
            # nn.Linear(64, 20)
            nn.Softmax(dim=1)
        )
        self.hetconv = HetConv(in_channels=inplanes, out_channels=outplanes, p=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.branch_weight = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.dwconv = nn.Sequential(
            nn.Conv2d(inner_dim, 2 * inner_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * inner_dim),
            nn.GELU()
        )

    def forward(self, fmap):
        # SPATIAL GLOBAL ATTENTION

        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q, k, v))

        # q = q* self.scale
        q = q * self.temperature  # 自适应缩放

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim=-1)
        # m_r = torch.ones_like(attn) *self.temperature1
        # attn = attn + torch.bernoulli(m_r) * -1e-12
        attn = self.dropkey(attn)

        out1 = einsum('b h i j, b h j d -> b h i d', attn, v)
        out1 = rearrange(out1, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        # out2_r =(self.avg_pool(fmap)).squeeze(-1).squeeze(-1)
        # out2_r =self.SPFC(out2_r).unsqueeze(-1).unsqueeze(-1)
        # out2 = self.hetconv(fmap)
        # out2 = out2 * out2_r

        out2 = self.dwconv(out1)
        out = out2

        return out

    def dropkey(selsf, attn):
        m_r = torch.ones_like(attn) * 0.3
        attn = attn + torch.bernoulli(m_r) * -1e-12
        return attn.cuda()


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


from Utils import SCA


class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()

    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=nn.BatchNorm2d):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # x = torch.squeeze(x,4)
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        # x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        # x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        # x = torch.unsqueeze(x,4)
        return x


class StripPooling(nn.Module):
    """
    Reference:
    """

    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels / 2)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels * 2, in_channels, 1, bias=False),
                                   norm_layer(in_channels))

        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out1 = self.conv3(torch.cat([x1, x2], dim=1))
        out2 = F.relu_(x + out1)
        return out2


def rel_to_abs(x):
    b, h, l, _, device, dtype = x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = repeat(logits, 'b h x y j -> b h x i y j', i=h)
    return logits


# positional embeddings

class AbsPosEmb(nn.Module):
    def __init__(
            self,
            fmap_size,
            dim_head
    ):
        super().__init__()
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(fmap_size, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(fmap_size, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits


class RelPosEmb(nn.Module):
    def __init__(
            self,
            fmap_size,
            dim_head
    ):
        super().__init__()
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(fmap_size * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(fmap_size * 2 - 1, dim_head) * scale)

    def forward(self, q):
        q = rearrange(q, 'b h (x y) d -> b h x y d', x=self.fmap_size)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=p, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


# dual-domain self-attention
class CrossATT(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim2=64,
            fmap_size,
            dim_out,
            heads=4,
            dim_head=128,
            rel_pos_emb=False,
            norm_layer=GroupNorm
    ):
        super().__init__()
        self.morm = GroupNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.temperature = nn.Parameter(torch.ones(dim_head))
        self.temperature1 = nn.Parameter(torch.ones(1))
        inner_dim = heads * dim_head
        norm_layer = nn.BatchNorm2d
        midplanes = outplanes = inplanes = 64
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_qkv_xl = nn.Conv2d(dim2, inner_dim * 3, 1, bias=False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, 1, bias=False)
        self.SPFC = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(128, 96),
            # mish(),
            nn.Linear(96, 64),
            # nn.Linear(64, 20)
            # nn.Softmax(dim = 1),
        )
        self.spconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, padding=(0, 0),
                      kernel_size=(1, 2), stride=(1, 1)),
            mish()
        )
        self.SPFC = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            # mish(),
            nn.Linear(32, 64),
            # nn.Linear(64, 20)
            nn.Softmax(dim=1)
        )
        self.hetconv = HetConv(in_channels=inplanes, out_channels=outplanes, p=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.branch_weight = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.dwconv = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim * 4, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(1,2*inner_dim),
            nn.BatchNorm2d(inner_dim * 4),
            nn.GELU())

        self.w1 = nn.Parameter(torch.rand(1))
        self.w2 = nn.Parameter(torch.rand(1))
        self.w3 = nn.Parameter(torch.rand(1))
        self.w4 = nn.Parameter(torch.rand(1))
        self.sca = SCA()

    def forward(self, *args):
        fmap, xl = args[0][0], args[0][1]

        # SPATIAL GLOBAL ATTENTION

        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q, k, v))

        q_l, k_l, v_l = self.to_qkv_xl(xl).chunk(3, dim=1)
        q_l, k_l, v_l = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q_l, k_l, v_l))
        v = (self.w1 * v + self.w2 * v_l)
        # q = (self.w3 * q + self.w4 * q_l)

        # q = q_l* self.scale
        q = q * self.temperature

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim=-1)

        attn = self.dropkey(attn)

        out1 = einsum('b h i j, b h j d -> b h i d', attn, v)
        out1 = rearrange(out1, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        # out2_r =(self.avg_pool(fmap)).squeeze(-1).squeeze(-1)
        # out2_r =self.SPFC(out2_r).unsqueeze(-1).unsqueeze(-1)
        # out2 = self.hetconv(fmap)
        # out2 = out2 * out2_r

        out2 = self.dwconv(out1)
        # out2 = self.sca(out2)

        return out2

    def dropkey(selsf, attn):
        m_r = torch.ones_like(attn) * 0.3
        attn = attn + torch.bernoulli(m_r) * -1e-12
        return attn


class IBCM(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim2=64,
            fmap_size,
            dim_out,
            proj_factor,
            downsample,
            heads=4,
            dim_head=128,
            rel_pos_emb=False,
            activation=nn.ReLU()
    ):
        super().__init__()

        # shortcut

        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(dim_out),
                activation,
                nn.Conv2d(dim_out, dim, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(dim),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion

        attention_dim = dim_out // proj_factor
        self.net1 = nn.Sequential(
            nn.Conv2d(dim, attention_dim, 1, bias=False),
            nn.BatchNorm2d(attention_dim),
            activation)
        self.net2 = nn.Sequential(
            nn.Conv2d(dim_out, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.net = nn.Sequential(
            # nn.Conv2d(dim, attention_dim, 1, bias=False),
            # nn.BatchNorm2d(attention_dim),
            # activation,
            CrossATT(
                dim=attention_dim,
                dim2=dim2,
                fmap_size=fmap_size,
                dim_out=dim_out,
                heads=heads,
                dim_head=dim_head,
                rel_pos_emb=rel_pos_emb),

            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(2 * attention_dim),
            activation,
            nn.Conv2d(2 * attention_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)

        )

        # init last batch norm gamma to zero

        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    def forward(self, *args):
        # print("每一次")
        x, xl = args[0][0], args[0][1]
        shortcut = self.shortcut(x)
        x = self.net1(x)
        out = self.net((x, xl))
        out += shortcut
        out = self.activation(out)
        # out = self.net2(out)
        # out = self.activation(out)

        return out, xl


class IBCMBlock(nn.Module):
    def __init__(
            self,
            *,
            dim=64,
            dim2=64,
            fmap_size,
            dim_out=64,
            proj_factor=6,
            num_layers=3,
            heads=2,
            dim_head=10,
            downsample=True,
            rel_pos_emb=False,
            activation=nn.ReLU()
    ):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            # print("dim :=" + str(is_first))
            # dim = (dim if is_first else dim_out)
            # print("dim out :=" + str(dim_out))
            layer_downsample = is_first and downsample
            layer_fmap_size = fmap_size // (2 if downsample and not is_first else 1)

            layers.append(IBCM(
                dim=dim,
                dim2=dim2,
                fmap_size=layer_fmap_size,  # 特征图大小
                dim_out=dim_out,  # 输出特征图通道
                proj_factor=proj_factor,  # 投影矩阵的因子
                heads=heads,  # 注意力头的数量
                dim_head=dim_head,  # 每个注意力头的维度
                downsample=layer_downsample,  # 是否下采样
                rel_pos_emb=rel_pos_emb,  # 是否位置编码
                activation=activation  # 激活函数类型
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, *args):
        x, xl = args[0][0], args[0][1]
        result, _ = self.net((x, xl))
        return torch.unsqueeze(result, dim=4)
