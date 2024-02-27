import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class U_Net(nn.Module):
    """
    输入：散斑G通道  (Batchsize, 1, 512, 512)
    输出：手写数字  (Batchsize, 1, 512, 512)
    """
    def __init__(self, in_channels=1, out_channels=1, features=4):
        super(U_Net, self).__init__()
        # self.norm = nn.InstanceNorm2d(8)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),  # 256
            nn.LeakyReLU(0.2),
        )
        self.down0 = Block(features, features, down=True, act = "leaky", use_dropout=False)     # 128    add for NLOS
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)   # 64
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False  # 32
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False  # 16
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False  # 8
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False  # 4
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False  # 2
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()  # 1
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.up8 = Block(features * 2, features, down=False, act="relu", use_dropout=False)
        self.up0 = Block(features * 2, features, down=False, act="relu", use_dropout=False)   # add for NLOS
        self.final_up = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x = self.norm(x)
        d0 = self.initial_down(x)
        d1 = self.down0(d0)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        up8 = self.up8(torch.cat([up7, d1], 1))
        up0 = self.up0(torch.cat([up8, d0], 1))
        return self.final_up(up0)


def test():
    # file_path = "F:\\FY4\\20211231_4000_025\\FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_20211231000000_20211231001459_4000M_V0001_30.DHF"
    # data_file = h5py.File(file_path, 'r')
    # input_data = data_file['longwave'][:]
    # input_data = torch.from_numpy(input_data)
    # input_data = input_data.view(-1, 8, 121, 121)
    # input_data = input_data.to(torch.float32)
    # model = Generator()
    # preds = model(input_data)
    # print(preds.shape)
    input_data = torch.randn((3, 1, 512, 512))
    model = U_Net()
    output_data = model(input_data)
    print(output_data.shape)
    # summary(model, input_size=(8, 256, 256), batch_size=1)


if __name__ == "__main__":
    test()