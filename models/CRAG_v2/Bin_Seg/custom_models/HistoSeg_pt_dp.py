import torch
import torch.nn as nn
import torch.nn.functional as F
import math


"""
Custom implementation of the HistoSeg's model mobilenet branch
incorporating "DEPooling" and Mish activation function

Still in testing phase

Implementation by Lorenz Rutkevich
"""


class QAU(nn.Module):
    def __init__(self, in_channels=32):
        super(QAU, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        f_a = x1
        f_b = self.conv(x2)
        f_act = self.sigmoid(f_b)
        f_add = f_act + f_a
        return f_add


class Conv2dSame(nn.Module):
    def __init__(
        self, filters, in_channels=1, stride=1, kernel_size=3, rate=1, padding=1
    ):
        super(Conv2dSame, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.rate = rate
        self.filters = filters
        self.conv = nn.Conv2d(
            self.in_channels,
            filters,
            kernel_size=kernel_size,
            stride=stride,
            dilation=rate,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        if self.stride != 1:
            kernel_size_effective = self.kernel_size + (self.kernel_size - 1) * (
                self.rate - 1
            )
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = nn.ZeroPad2d((pad_beg, pad_end))(x)
        x = self.conv(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _check_power_of_two(x, inputs=None):
    def log2(x):
        return math.log(x, 2)

    if inputs is not None:
        x_h, x_w = x.size()[2:]
        inputs_h, inputs_w = inputs.size()[2:]
        x_h = int(x_h)
        x_w = int(x_w)
        inputs_h = int(inputs_h)
        inputs_w = int(inputs_w)
        if (
            math.ceil(log2(x_h)) != math.floor(log2(x_h))
            or math.ceil(log2(x_w)) != math.floor(log2(x_w))
            or math.ceil(log2(inputs_h)) != math.floor(log2(inputs_h))
            or math.ceil(log2(inputs_w)) != math.floor(log2(inputs_w))
        ):
            nearest_p_o_2 = 2 ** int(math.ceil(log2(max(x_h, x_w, inputs_h, inputs_w))))
            x = nn.ZeroPad2d((0, nearest_p_o_2 - x_w, 0, nearest_p_o_2 - x_h))(x)
            inputs = nn.ZeroPad2d(
                (0, nearest_p_o_2 - inputs_w, 0, nearest_p_o_2 - inputs_h)
            )(inputs)
            return x, inputs
        else:
            return x, inputs
    else:
        x_h, x_w = x.size()[2:]
        x_h = int(x_h)
        x_w = int(x_w)
        if math.ceil(log2(x_h)) != math.floor(log2(x_h)) or math.ceil(
            log2(x_w)
        ) != math.floor(log2(x_w)):
            nearest_p_o_2 = 2 ** int(math.ceil(log2(max(x_h, x_w))))
            x = nn.ZeroPad2d((0, nearest_p_o_2 - x_w, 0, nearest_p_o_2 - x_h))(x)
            return x
        else:
            return x


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        expansion,
        stride,
        alpha,
        filters,
        block_id,
        skip_connection,
        rate=1,
    ):
        super(InvertedResidualBlock, self).__init__()

        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

        self.block_id = block_id
        self.skip_connection = skip_connection

        self.conv1 = nn.Conv2d(
            in_channels, in_channels * expansion, kernel_size=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels * expansion, momentum=0.999, eps=1e-3)
        self.mish1 = nn.Mish()

        self.conv2 = nn.Conv2d(
            in_channels * expansion,
            in_channels * expansion,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=rate,
            bias=False,
            groups=in_channels * expansion,
            # groups must be defined to achieve depthwise convolution, else the model parameters will explode
            # ~2m -> ~42m
        )
        self.bn2 = nn.BatchNorm2d(in_channels * expansion, momentum=0.999, eps=1e-3)
        self.mish2 = nn.Mish()

        self.conv3 = nn.Conv2d(
            in_channels * expansion,
            pointwise_filters,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(pointwise_filters, momentum=0.999, eps=1e-3)

        if skip_connection:
            if not in_channels * expansion == pointwise_filters:
                self.conv4 = nn.Conv2d(
                    in_channels, pointwise_filters, kernel_size=1, padding=0, bias=False
                )
                self.bn4 = nn.BatchNorm2d(pointwise_filters, momentum=0.999, eps=1e-3)

    def forward(self, inputs):
        x = inputs
        if self.block_id:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.mish1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mish2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.skip_connection:
            #     if not x.shape == inputs.shape:
            #        inputs = self.conv4(inputs)
            #       inputs = self.bn4(inputs)

            x, inputs = _check_power_of_two(x, inputs)
            return inputs + x

        x = _check_power_of_two(x)
        return x


class SimpleConv(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        use_activation=True,
        eps=1e-3,
        activation=nn.Mish(inplace=True),
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.eps = eps
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_activation = use_activation
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=self.bias,
        )
        self.bn = nn.BatchNorm2d(self.out_channels, momentum=0.9997, eps=self.eps)
        self.activation = activation

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
        eps=1e-3,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.eps = eps
        self.pooling_out = pooling_out
        conv = SimpleConv(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.bias,
            eps=self.eps,
            activation=nn.Mish(inplace=True),
        )
        self.conv = nn.Sequential(nn.AdaptiveAvgPool2d(self.pooling_out), conv)

    def forward(self, x):
        x = self.conv(x)
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


class Add(nn.Module):
    def __init__(self, add_modules: list = []):
        super().__init__()
        self.add_modules = add_modules

    def forward(self, x):
        tensors = [module(x) for module in self.add_modules]
        x = sum(tensors)
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


class HistoSeg(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        classes: int = 1,
        OS: int = 8,
        alpha: float = 1.0,
        dropout_rate: float = 0.1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        assert img_size % 8 == 0

        self.img_size = img_size
        self.in_channels = in_channels
        self.classes = classes
        self.OS = OS
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.depths_1 = [128, 128, 128]
        self.depths_2 = [256, 256, 256]
        self.depths_3 = [728, 728, 728]
        self.depths_4 = [728, 1024, 1024]
        self.depths_5 = [1536, 1536, 2048]

        self.first_block_filters = _make_divisible(32 * self.alpha, 8)
        self.conv1 = SimpleConv(
            self.in_channels,
            self.first_block_filters,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False,
        )
        self.block1 = InvertedResidualBlock(
            self.first_block_filters,
            expansion=1,
            stride=1,
            alpha=self.alpha,
            filters=16,
            block_id=0,
            skip_connection=False,
        )
        self.block2 = InvertedResidualBlock(
            16,
            expansion=6,
            stride=2,
            alpha=self.alpha,
            filters=24,
            block_id=1,
            skip_connection=False,
        )
        self.block3 = InvertedResidualBlock(
            24,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=24,
            block_id=2,
            skip_connection=True,
        )
        self.block4 = InvertedResidualBlock(
            24,
            expansion=6,
            stride=2,
            alpha=self.alpha,
            filters=32,
            block_id=3,
            skip_connection=False,
        )

        self.block5 = InvertedResidualBlock(
            32,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=32,
            block_id=4,
            skip_connection=True,
        )
        self.block6 = InvertedResidualBlock(
            32,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=32,
            block_id=5,
            skip_connection=True,
        )

        self.block6_pool = DEPooling(
            32,
            32,
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
        )

        self.x_1 = QAU()

        self.block7 = InvertedResidualBlock(
            32,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=64,
            block_id=6,
            skip_connection=False,
        )
        self.block8 = InvertedResidualBlock(
            64,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=64,
            block_id=7,
            skip_connection=True,
            rate=2,
        )
        self.block9 = InvertedResidualBlock(
            64,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=64,
            block_id=8,
            skip_connection=True,
            rate=2,
        )
        self.block10 = InvertedResidualBlock(
            64,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=64,
            block_id=9,
            skip_connection=True,
            rate=2,
        )

        self.block10_pool = DEPooling(
            64,
            64,
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
        )
        self.block11 = InvertedResidualBlock(
            64,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=96,
            block_id=10,
            skip_connection=False,
            rate=2,
        )
        self.block12 = InvertedResidualBlock(
            96,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=96,
            block_id=11,
            skip_connection=True,
            rate=2,
        )
        self.block13 = InvertedResidualBlock(
            96,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=96,
            block_id=12,
            skip_connection=True,
            rate=2,
        )

        self.block13_pool = DEPooling(
            96,
            96,
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
        )

        self.block14 = InvertedResidualBlock(
            96,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=160,
            block_id=13,
            skip_connection=False,
            rate=2,
        )
        self.block15 = InvertedResidualBlock(
            160,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=160,
            block_id=14,
            skip_connection=True,
            rate=4,
        )
        self.block16 = InvertedResidualBlock(
            160,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=160,
            block_id=15,
            skip_connection=True,
            rate=4,
        )

        self.block16_pool = DEPooling(
            160,
            160,
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
        )

        self.block17 = InvertedResidualBlock(
            160,
            expansion=6,
            stride=1,
            alpha=self.alpha,
            filters=320,
            block_id=16,
            skip_connection=False,
            rate=4,
        )

        self.block17_pool = DEPooling(
            320,
            320,
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
        )

        self.aap = SimpleConvAAP(
            320,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            pooling_out=(1, 1),
            eps=1e-5,
        )

        self.up1 = Up()
        self.bconv1 = SimpleConv(
            320,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            eps=1e-5,
        )

        self.bconv2 = SimpleConv(
            512,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            use_activation=False,
            eps=1e-5,
        )

        self.poolconv = SimpleConv(
            608,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            use_activation=False,
        )
        self.poolup = Up()
        self.x_q_au = QAU()
        self.x_1_1 = nn.Conv2d(256, 256, kernel_size=1, padding="same")
        self.z = Add([self.x_q_au, self.x_1_1])
        self.bmish1 = nn.ReLU(inplace=True)
        self.dr = nn.Dropout(self.dropout_rate)
        self.conv_down = nn.Conv2d(
            512, 256, kernel_size=1, stride=1, padding="same", bias=False
        )

        self.conv_last = nn.Conv2d(
            256, self.classes, kernel_size=1, stride=1, padding="same", bias=False
        )
        self.up2 = Up()
        self.activation = nn.Sigmoid() if self.classes == 1 else nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        b6_pool = self.block6_pool(x)
        x_1 = self.x_1(x, x)
        x = self.block7(x_1)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        b13_pool = self.block13_pool(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        b16_pool = self.block16_pool(x)
        x = self.block17(x)
        b17_pool = self.block17_pool(x)
        self.shape_before_pool = x.shape
        p1 = torch.cat([b6_pool, b13_pool, b16_pool, b17_pool], dim=1)
        p1 = self.poolconv(p1)
        p1 = self.poolup(p1, self.shape_before_pool[2], self.shape_before_pool[3])
        b4 = self.aap(x)
        b4 = self.up1(b4, self.shape_before_pool[2], self.shape_before_pool[3])
        b0 = self.bconv1(x)

        x = torch.cat([b4, b0], dim=1)
        x = self.bconv2(x)
        x = torch.cat([x, p1], dim=1)
        x = self.dr(x)
        x = self.conv_down(x)
        x = self.conv_last(x)
        x = self.up2(x, input.shape[2], input.shape[3])
        x = self.activation(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    # for mobilenetv2 img_size must be 128 or bigger
    # batch_size smaller than 2 won't work with self.aap()
    model = HistoSeg(img_size=256, in_channels=3, classes=1, OS=16, alpha=1.0)
    y = model(x)
    print(y.shape)
    import thop

    flops, params = thop.profile(model, inputs=(x,))
    print(flops, params)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)
    from torchsummary import summary

    summary(model, (3, 256, 256))
