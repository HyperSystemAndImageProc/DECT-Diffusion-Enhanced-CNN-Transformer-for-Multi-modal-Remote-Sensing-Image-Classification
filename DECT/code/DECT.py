import torch
from einops import rearrange
from torch import nn
from SPAI import SPAI
from Utils import SCA
from PAT import PoolFormerBlock
from IBCMBlock import IBCMBlock, BottleStack


class DEMTnet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            num_classes=6,
            num_tokens=4,
            dim=64,
            emb_dropout=0.1,
            num_patches=1,
            patch=13

    ):
        super(DEMTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv3d_hy1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.conv3d_dh1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.conv3d_hy2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv3d_dh2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2d_li1 = nn.Sequential(
            nn.Conv2d(in_channels=num_patches, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2d_li2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),  # 原16
            nn.ReLU(),
        )
        self.conv2d_li3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),  # 原16
            nn.ReLU(),
        )
        self.conv2d_dl1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2d_dl2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),  # 原16
            nn.ReLU(),
        )

        self.fc_token1 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(81, 64),
            nn.Softmax()
        )
        self.fc_token2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(81, 32),
            nn.Softmax()
        )
        self.fc_token3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(81, 16),
            nn.Softmax()
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * 28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=16 * 28, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2d_hy1 = nn.Sequential(
            nn.Conv2d(in_channels=8 * 28, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),  # 原64
            nn.ReLU(),
        )
        self.conv2d_dy1 = nn.Sequential(
            nn.Conv2d(in_channels=8 * 28, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),  # 原64
            nn.ReLU(),
        )
        self.conv2d_hy2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2d_hy3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2d_features2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2d_mix1 = nn.Sequential(
            nn.Conv2d(96, out_channels=32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2d_mix2 = nn.Sequential(
            nn.Conv2d(32, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2d_mix3 = nn.Sequential(
            nn.Conv2d(32, out_channels=96, kernel_size=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.to_cls_token = nn.Identity()
        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.transformer_h = BottleStack(dim=128, fmap_size=9, dim_out=128, proj_factor=2, downsample=False, heads=4,
                                         dim_head=16, num_layers=3, rel_pos_emb=False, activation=nn.ReLU())
        self.transformer_d = BottleStack(dim=64, fmap_size=9, dim_out=128, proj_factor=2, downsample=False, heads=4,
                                         dim_head=16, num_layers=3, rel_pos_emb=False, activation=nn.ReLU())
        self.transformer_l = BottleStack(dim=16, fmap_size=9, dim_out=128, proj_factor=2, downsample=False, heads=4,
                                         dim_head=16, num_layers=3, rel_pos_emb=False, activation=nn.ReLU())
        self.transformer_dl = BottleStack(dim=64, fmap_size=9, dim_out=128, proj_factor=2, downsample=False, heads=4,
                                          dim_head=16, num_layers=3, rel_pos_emb=False, activation=nn.ReLU())
        self.transformer_dual = IBCMBlock(dim=128, dim2=16, fmap_size=patch - 2, dim_out=256, proj_factor=2,
                                          downsample=False,
                                          heads=4,
                                          dim_head=16, num_layers=3, rel_pos_emb=False, activation=nn.ReLU())
        self.transformer_dual2 = IBCMBlock(dim=64, dim2=16, fmap_size=9, dim_out=128, proj_factor=1,
                                           downsample=False,
                                           heads=4,
                                           dim_head=16, num_layers=3, rel_pos_emb=False, activation=nn.ReLU())
        self.GP = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Sequential(nn.Dropout(p=0.3),
                                nn.Linear(256, num_classes)
                                )
        self.FC2 = nn.Sequential(nn.Dropout(p=0.2),
                                 nn.Linear(128, num_classes)
                                 )
        self.FC3 = nn.Sequential(nn.Linear(128, num_classes))

        self.beta = nn.Parameter(torch.tensor(0.5))
        self.w1 = nn.Parameter(torch.rand(1))
        self.w2 = nn.Parameter(torch.rand(1))
        self.w3 = nn.Parameter(torch.rand(1))

        self.acsa = SPAI(dim=128, num_heads=16)
        self.acsa2 = SPAI(dim=128, num_heads=16)
        self.model1 = PoolFormerBlock(dim=30)
        self.model2 = PoolFormerBlock(dim=30)
        self.sca = SCA()

    # 384
    def forward(self, x1, xD, x2):
        # Lidar支路
        # x_l_1 = self.conv2d_li1(torch.squeeze(x2))  # this is augsburg
        x_l_1 = self.conv2d_li1(x2)  # 64 32 9 9
        x_l_2 = self.conv2d_li2(x_l_1)  # 64 16 9 9

        x2 = torch.unsqueeze(x2, dim=2)
        yl = x2.expand_as(x1)
        # -------Augsburg------------
        # sliced_x2 = x2[:, :, :2, :, :]
        # x2 = x2.repeat(1, 1, 7, 1, 1)
        # x2 = torch.cat((x2, sliced_x2), dim=2)
        # yl = x2.expand_as(x1)
        # -------Augsburg------------

        x_h_1 = self.conv3d_hy1(x1)
        x_h_2 = self.conv3d_hy2(x_h_1)
        x_h_3 = rearrange(x_h_2, 'b c h w y ->b (c h) w y')
        x_h_4 = self.conv2d_hy1(x_h_3)  # 64 64 9 9
        x_h_4 = self.acsa(x_h_4)  # SPAI
        x_h_5 = torch.unsqueeze(x_h_4, dim=4)
        x_h_6 = self.transformer_dual((torch.squeeze(x_h_5), torch.squeeze(x_l_2)))  # 64 128 9 9 1
        x_h_6 = torch.squeeze(x_h_6, dim=4)  # 64 128 9 9  (N, C, H, W, 1) 转换为 (N, C, H, W)

        xDs = self.beta * xD + (1 - self.beta) * yl
        x_d_1 = self.conv3d_dh1(xDs)
        x_d_2 = self.conv3d_dh2(x_d_1)
        x_d_3 = rearrange(x_d_2, 'b c h w y ->b (c h) w y')
        x_d_4 = self.conv2d_dy1(x_d_3)
        x_d_4 = self.acsa(x_d_4)  # SPAI
        x_d_5 = torch.unsqueeze(x_d_4, dim=4)
        x_d_6 = self.transformer_dual((torch.squeeze(x_d_5), torch.squeeze(x_l_2)))
        x_d_6 = torch.squeeze(x_d_6, dim=4)  # (N, C, H, W, 1) 转换为 (N, C, H, W)

        # 自适应特征融合
        out = (self.w1 * x_h_6 + self.w2 * x_d_6)
        # out = x_h_6  # only HSI
        out = self.GP(out)  # 64 384 1 1
        out = out.squeeze(2).squeeze(2)  # 64 384
        out = self.FC3(out)

        return out


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DEMTnet(patch=11).to(device)
    model.eval()
    input1 = torch.randn(64, 1, 30, 11, 11).to(device)
    inputD = torch.randn(64, 1, 30, 11, 11).to(device)
    input2 = torch.randn(64, 1, 11, 11).to(device)
    model(input1, inputD, input2)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(model, (input1, inputD, input2))
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, (input1, inputD, input2))

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")

    print(flop_count_table(flops, max_depth=1))
