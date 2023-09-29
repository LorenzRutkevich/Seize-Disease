import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as summary

"""
Custom model for semantic segmentation using mixed blocks and "DEPooling" layers.
Still in testing phase.

Implemented by: Lorenz Rutkevich
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
        dilation=1,
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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x, h, w):
        x = nn.Upsample(size=(h, w), mode="bilinear", align_corners=True)(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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
    ):
        super(ResConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.padding = padding

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
            out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding=0
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
        # print("x", x.shape)
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
    ) -> None:
        super(SimpleConv, self).__init__()
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
        x = self.activation(x)
        return x


class CustomNet(nn.Module):
    def __init__(self, input_shape, classes, dropout) -> None:
        super(CustomNet, self).__init__()
        self.input_shape = input_shape
        self.classes = classes
        self.dropout = dropout
        """
        input_shape: (height, width, channels)
        classes: number of classes
        dropout: dropout rate
        """

        self.sepconv_1 = SepConvBN(
            3, 32, stride=1, kernel_size=3, rate=1, depth_activation=True, epsilon=1e-3
        )
        self.sepconv_2 = SepConvBN(
            32, 64, stride=1, kernel_size=3, rate=1, depth_activation=True, epsilon=1e-3
        )
        self.sepconv_3 = SepConvBN(
            64,
            128,
            stride=1,
            kernel_size=3,
            rate=1,
            depth_activation=True,
            epsilon=1e-3,
        )

        self.att_1 = EfficientAttention(128, 128, 4, 128)
        self.res_1 = ResConv(128, 128, stride=1, kernel_size=3, rate=1, epsilon=1e-3)
        self.depool_1 = DEPooling(
            128,
            128,
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
        self.res_2 = ResConv(128, 128, stride=1, kernel_size=3, rate=1, epsilon=1e-3)
        self.depool_2 = DEPooling(
            128,
            128,
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
        self.sepconv_1_0 = SepConvBN(
            3, 32, stride=1, kernel_size=3, rate=1, depth_activation=True, epsilon=1e-3
        )
        self.sepconv_2_0 = SepConvBN(
            32, 64, stride=1, kernel_size=3, rate=1, depth_activation=True, epsilon=1e-3
        )
        self.sepconv_3_0 = SepConvBN(
            64,
            128,
            stride=1,
            kernel_size=3,
            rate=1,
            depth_activation=True,
            epsilon=1e-3,
        )

        self.att_1_0 = EfficientAttention(128, 128, 4, 128)
        self.res_1_0 = ResConv(128, 128, stride=1, kernel_size=3, rate=1, epsilon=1e-3)
        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.up1 = Up()
        self.saap = SimpleConv(
            128, 128, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )
        self.up2 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.cat = Concatenate()
        self.res_up = ResConv(256, 256, stride=1, kernel_size=3, rate=1, epsilon=1e-3)
        self.sepconv_up = SepConvBN(
            256,
            128,
            stride=1,
            kernel_size=3,
            rate=1,
            depth_activation=True,
            epsilon=1e-3,
        )
        self.conv_up = nn.Conv2d(
            128, 64, kernel_size=1, stride=1, bias=False, padding=0
        )
        self.conv_up_1 = nn.Conv2d(
            64, 32, kernel_size=1, stride=1, bias=False, padding=0
        )
        self.conv_up_2 = nn.Conv2d(
            32, 16, kernel_size=1, stride=1, bias=False, padding=0
        )
        self.upsample_1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample_2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.up_ = Up()
        self.out = nn.Conv2d(
            16, self.classes, kernel_size=1, stride=1, bias=False, padding=0
        )
        self.out_activation = nn.Sigmoid() if self.classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        xx = x

        # stage 1
        x = self.sepconv_1(x)
        x = self.sepconv_2(x)
        x = self.sepconv_3(x)
        x = self.att_1(x)
        x = self.res_1(x)
        x = self.depool_1(x)
        x = self.res_2(x)
        x = self.depool_2(x)

        # stage 2
        x1 = self.sepconv_1_0(xx)
        x1 = self.sepconv_2_0(x1)
        x1 = self.sepconv_3_0(x1)
        x1 = self.att_1_0(x1)
        x1 = self.res_1_0(x1)
        x1 = self.aap(x1)
        x1 = self.up1(x1, 32, 32)
        x1 = self.saap(x1)
        x1 = self.up2(x1)

        # feature concatenation
        x = torch.cat([x, x1], dim=1)

        # dropout
        x = nn.Dropout2d(self.dropout)(x)

        # stage 3
        x = self.res_up(x)
        x = self.sepconv_up(x)
        x = self.conv_up(x)
        x = self.conv_up_1(x)
        x = self.conv_up_2(x)

        # output stage
        x = self.up_(x, xx.shape[2], xx.shape[3])
        x = self.out(x)
        x = self.out_activation(x)
        return x


if __name__ == "__main__":
    rand = torch.randn(1, 3, 256, 256)
    model = CustomNet(input_shape=(256, 256, 3), classes=32, dropout=0.5)
    out = model(rand)
    print(out.shape)
    print(summary.summary(model, (3, 256, 256)))
