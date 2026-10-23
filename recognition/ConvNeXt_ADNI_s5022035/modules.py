import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Preserve the earlier class name for any existing checkpoints or imports.
BaselineNN = BaselineCNN


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        
    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = x + input_x
        return x


class ConvNeXtTiny(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        depths=(3, 1, 3, 1),
        dims=(96, 192, 384, 768),
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                    LayerNorm2d(dims[0]),
                )
            ]
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
                )
                for i in range(4)
            ]
        )
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)

        x = x.mean([-2, -1])
        x = self.norm(x)
        return self.head(x)


def build_torchvision_convnext_tiny(pretrained=False, num_classes=2):
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)

    first_conv = model.features[0][0]
    grayscale_conv = nn.Conv2d(
        1,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    )
    with torch.no_grad():
        grayscale_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        if first_conv.bias is not None:
            grayscale_conv.bias.copy_(first_conv.bias)
    model.features[0][0] = grayscale_conv
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model