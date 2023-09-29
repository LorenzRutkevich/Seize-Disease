import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as summary

"""
Custom Neural Network for semantic segmentation
still in testing phase

implemented by: Lorenz Rutkevich
"""


def test_backbone(backbone, input_shape=(256, 256, 3), n_classes=3):
    out = backbone(torch.randn(1, input_shape[2], input_shape[0], input_shape[1]))
    print(out.shape)


class SepConvBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel_size=3,
        rate=1,
        depth_activation=False,
        epsilon=1e-3,
    ):
        super(SepConvBN, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth_activation = depth_activation
        self.epsilon = epsilon
        self.groups = self.gcd(self.in_channels, self.out_channels)
        if self.stride == 1:
            padding = "same"
        else:
            padding = "valid"
        self.depthwise_conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=rate,
            bias=False,
            padding=padding,
            groups=self.groups,
        )
        self.pointwise_conv2d = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channels, eps=epsilon)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=epsilon)

    def gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def forward(self, x):
        if self.stride != 1:
            kernel_size_effective = self.kernel_size + (self.kernel_size - 1) * (
                self.rate - 1
            )
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = nn.ZeroPad2d((pad_beg, pad_end, pad_beg, pad_end))(x)
        if not self.depth_activation:
            x = nn.Mish()(x)
        x = self.depthwise_conv2d(x)
        x = self.bn1(x)
        if self.depth_activation:
            x = nn.Mish()(x)
        x = self.pointwise_conv2d(x)
        x = self.bn2(x)
        if self.depth_activation:
            x = nn.Mish()(x)
        return x


class DEPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel_size=3,
        rate=1,
        activation=False,
        epsilon=1e-3,
        dilation=2,
        pooling_size=2,
        pooling_stride=2,
        padding_conv=0,
        padding_pool=0,
    ) -> None:
        super(DEPooling, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.epsilon = epsilon
        self.dilation = dilation
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.padding_conv = padding_conv
        self.padding_pool = padding_pool

        self.dilated_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=rate,
            bias=False,
            padding=padding_conv,
        )

        self.bn1 = nn.BatchNorm2d(out_channels, eps=epsilon)

        self.pooling = nn.MaxPool2d(
            kernel_size=pooling_size,
            stride=pooling_stride,
            padding=padding_pool,
        )

        if self.activation:
            self.activation = nn.Mish()

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.bn1(x)
        x = self.pooling(x)
        if self.activation:
            x = self.activation(x)
        return x


class Concatenate(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, concat_list=[]):
        x = torch.cat(concat_list, dim=1)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = SimpleConv(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = SimpleConv(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel_size=3,
        rate=1,
        epsilon=1e-3,
        padding=1,
        return_skip=False,
    ):
        super(ResConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.padding = padding
        self.return_skip = return_skip

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=rate,
            bias=False,
            padding=padding,
        )

        self.bn1 = nn.BatchNorm2d(out_channels, eps=epsilon)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            padding=0 if padding == 1 else 1,
        )

        self.bn2 = nn.BatchNorm2d(out_channels, eps=epsilon)

        self.activation = nn.Mish()

    def forward(self, x):
        res = x
        # print("res", res.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        if self.return_skip:
            return x, res
        x = x + res
        return x


class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(
                keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2
            )
            query = F.softmax(
                queries[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )
            value = values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(
                n, head_value_channels, h, w
            )
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class SimpleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        padding_mode="zeros",
        use_activation=True,
    ) -> None:
        super(SimpleConv, self).__init__()
        self.use_activation = use_activation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class SimpleConvAAP(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        pooling_out=(1, 1),
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pooling_out = pooling_out
        conv = SimpleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv = nn.Sequential(nn.AdaptiveAvgPool2d(self.pooling_out), conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class Add(nn.Module):
    def __init__(self, add_modules: list = []):
        super().__init__()
        self.add_modules = add_modules

    def forward(self, x):
        tensors = [module(x) for module in self.add_modules]
        x = sum(tensors)
        return x


class SimpleUp(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x, h, w):
        x = nn.Upsample(size=(h, w), mode="bilinear", align_corners=True)(x)
        return x


class CustomNet(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        classes: int,
        dropout: int = 0.0,
    ) -> None:
        super(CustomNet, self).__init__()
        self.type = type
        self.input_shape = input_shape
        self.classes = classes
        self.dropout = dropout
        filters = [32, 64, 128, 256, 512, 1024, 1536]
        atrous_rates = (6, 12, 18)

        # stage 1
        self.conv1 = SimpleConv(
            input_shape[2], filters[0], kernel_size=3, stride=1, padding=1
        )
        self.conv2 = SimpleConv(
            filters[0], filters[1], kernel_size=3, stride=1, padding=0
        )
        self.conv3 = SimpleConv(
            filters[1], filters[2], kernel_size=3, stride=2, padding=0
        )
        self.conv4 = ResConv(
            filters[2], filters[2], stride=1, kernel_size=3, rate=1, epsilon=1e-3
        )
        self.conv5 = SimpleConv(
            filters[2], filters[3], kernel_size=3, stride=2, padding=1
        )
        self.conv6 = ResConv(
            filters[3], filters[3], stride=1, kernel_size=3, rate=1, epsilon=1e-3
        )
        self.pool1 = DEPooling(
            filters[3],
            filters[3],
            stride=1,
            kernel_size=3,
            rate=1,
            activation=True,
            epsilon=1e-3,
            dilation=2,
            pooling_size=2,
            pooling_stride=2,
            padding_conv=1,
            padding_pool=0,
        )
        self.attention1 = EfficientAttention(
            filters[3], filters[3] // 2, 4, filters[3] // 2
        )
        # sep 1
        self.sep1 = SepConvBN(
            filters[3], filters[3], kernel_size=3, stride=1, rate=atrous_rates[0]
        )

        # stage 2
        self.conv1_1 = ResConv(
            filters[3], filters[3], stride=1, kernel_size=3, rate=1, epsilon=1e-3
        )
        self.conv2_1 = ResConv(
            filters[3], filters[3], stride=1, kernel_size=3, rate=1, epsilon=1e-3
        )
        self.conv3_1 = ResConv(
            filters[3], filters[3], stride=1, kernel_size=3, rate=1, epsilon=1e-3
        )
        self.pool2 = DEPooling(
            filters[3],
            filters[3],
            stride=1,
            kernel_size=3,
            rate=1,
            activation=True,
            epsilon=1e-3,
            dilation=2,
            pooling_size=2,
            pooling_stride=1,
            padding_conv=1,
            padding_pool=1,
        )
        self.attention2 = EfficientAttention(
            filters[3], filters[3] // 2, 4, filters[3] // 2
        )
        # sep 2
        self.sep2 = SepConvBN(
            filters[3], filters[3], kernel_size=3, stride=1, rate=atrous_rates[1]
        )

        # stage 3
        self.conv1_2 = SimpleConv(
            filters[3], filters[4], kernel_size=3, stride=1, padding=1
        )
        self.conv2_2 = ResConv(
            filters[4], filters[4], stride=1, kernel_size=3, rate=1, epsilon=1e-3
        )
        self.conv3_2 = ResConv(
            filters[4], filters[4], stride=1, kernel_size=3, rate=1, epsilon=1e-3
        )
        self.conv4_2 = ResConv(
            filters[4], filters[4], stride=1, kernel_size=3, rate=1, epsilon=1e-3
        )
        self.pool3 = DEPooling(
            filters[4],
            filters[4],
            stride=1,
            kernel_size=3,
            rate=1,
            activation=True,
            epsilon=1e-3,
            dilation=2,
            pooling_size=2,
            pooling_stride=2,
            padding_conv=0,
            padding_pool=1,
        )
        self.attention3 = EfficientAttention(
            filters[4], filters[4] // 2, 4, filters[4] // 2
        )
        # sep 3
        self.sep3 = SepConvBN(
            filters[4], filters[4], kernel_size=3, stride=1, rate=atrous_rates[2]
        )

        # feature collection
        self.pool = SimpleConvAAP(
            filters[4],
            filters[4],
            kernel_size=3,
            stride=1,
            padding=1,
            pooling_out=(1, 1),
        )
        self.pool_up = SimpleUp()
        self.s3_up = SimpleUp()

        # out branch
        self.conv_d0 = SimpleConv(
            filters[6], filters[5], kernel_size=3, stride=1, padding=1
        )
        self.up_out1 = SimpleUp()
        self.conv_d1 = SimpleConv(
            filters[5], filters[4], kernel_size=3, stride=1, padding=1
        )
        self.att_d1 = EfficientAttention(
            filters[4], filters[4] // 2, 4, filters[4] // 2
        )
        self.up_out2 = SimpleUp()
        self.conv_last = SimpleConv(
            filters[4],
            filters[3],
            kernel_size=1,
            stride=1,
            padding=0,
            use_activation=False,
        )

        self.dr = nn.Dropout2d(p=self.dropout)

        self.conv_out = nn.Conv2d(
            filters[3], self.classes, kernel_size=1, stride=1, padding=0
        )
        self.activation = nn.Sigmoid() if self.classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        shape_in = x.shape
        b1 = self.conv1(x)
        b1 = self.conv2(b1)
        b1 = self.conv3(b1)
        b1 = self.conv4(b1)
        b1 = self.conv5(b1)
        b1 = self.conv6(b1)
        b1 = self.pool1(b1)
        b1 = self.attention1(b1)
        sep1 = self.sep1(b1)
        sep1 = F.pad(sep1, (0, 1, 0, 1))

        b2 = self.conv1_1(b1)
        b2 = self.conv2_1(b2)
        b2 = self.conv3_1(b2)
        b2 = self.pool2(b2)
        b2 = self.attention2(b2)
        sep2 = self.sep2(b2)

        b3 = self.conv1_2(b2)
        b3 = self.conv2_2(b3)
        b3 = self.conv3_2(b3)
        b3 = self.conv4_2(b3)
        b3 = self.pool3(b3)
        b3 = self.pool3(b3)
        b3 = self.attention3(b3)
        t_h, t_w = b2.shape[2], b2.shape[3]

        sep3 = self.sep3(b3)
        sep3 = self.s3_up(sep3, t_h, t_w)
        b3 = self.s3_up(b3, t_h, t_w)

        b4 = self.pool(b3)
        b4 = self.pool_up(b4, t_h, t_w)

        x = torch.cat([sep1, sep2, sep3, b4], dim=1)

        x = self.conv_d0(x)
        x = self.up_out1(x, t_h * 2, t_w * 2)
        x = self.conv_d1(x)
        x = self.att_d1(x)
        x = self.conv_last(x)
        x = self.dr(x)
        x = self.conv_out(x)
        x = self.up_out2(x, shape_in[2], shape_in[3])
        x = self.activation(x)
        return x


if __name__ == "__main__":
    rand = torch.randn(2, 3, 256, 256)
    model = CustomNet(input_shape=(256, 256, 3), classes=1, dropout=0.0)
    out = model(rand)
    print(out.shape, rand.shape)
    print(out.shape)
    print(summary.summary(model, (3, 256, 256)))
