from .resnet import resnet38, resnet110, resnet116, resnet14x2, resnet38x2, resnet110x2
from .resnet import resnet8x4, resnet14x4, resnet32x4, resnet38x4
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg8_bn, vgg13_bn
from .mobilenetv2 import mobile_half, mobile_half_double
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_1_5

from .resnet_imagenet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from .resnet_imagenet import wide_resnet10_2, wide_resnet18_2, wide_resnet34_2
from .mobilenetv2_imagenet import mobilenet_v2
from .shuffleNetv2_imagenet import shufflenet_v2_x1_0

model_dict = {
    'resnet38_100': resnet38,
    'resnet110_100': resnet110,
    'resnet116_100': resnet116,
    'resnet14x2_100': resnet14x2,
    'resnet38x2_100': resnet38x2,
    'resnet110x2_100': resnet110x2,
    'resnet8x4_100': resnet8x4,
    'resnet14x4_100': resnet14x4,
    'resnet32x4_100': resnet32x4,
    'resnet38x4_100': resnet38x4,
    'vgg8_100': vgg8_bn,
    'vgg13_100': vgg13_bn,
    'MobileNetV2_100': mobile_half,
    'MobileNetV2_1_0_100': mobile_half_double,
    'ShuffleV1_100': ShuffleV1,
    'ShuffleV2_100': ShuffleV2,
    'ShuffleV2_1_5_100': ShuffleV2_1_5,
    'ResNet50_100': ResNet50,
    'wrn_16_1_100': wrn_16_1,
    'wrn_16_2_100': wrn_16_2,
    'wrn_40_1_100': wrn_40_1,
    'wrn_40_2_100': wrn_40_2,


    'resnet38_200': resnet38,
    'resnet110_200': resnet110,
    'resnet116_200': resnet116,
    'resnet14x2_200': resnet14x2,
    'resnet38x2_200': resnet38x2,
    'resnet110x2_200': resnet110x2,
    'resnet8x4_200': resnet8x4,
    'resnet14x4_200': resnet14x4,
    'resnet32x4_200': resnet32x4,
    'resnet38x4_200': resnet38x4,
    'vgg8_200': vgg8_bn,
    'vgg13_200': vgg13_bn,
    'MobileNetV2_200': mobile_half,
    'MobileNetV2_1_0_200': mobile_half_double,
    'ShuffleV1_200': ShuffleV1,
    'ShuffleV2_200': ShuffleV2,
    'ShuffleV2_1_5_200': ShuffleV2_1_5,
    'ResNet50_200': ResNet50,
    'wrn_16_1_200': wrn_16_1,
    'wrn_16_2_200': wrn_16_2,
    'wrn_40_1_200': wrn_40_1,
    'wrn_40_2_200': wrn_40_2,
    
    'ResNet18_1000': resnet18,
    'ResNet34_1000': resnet34,
    'ResNet50_1000': resnet50,
    'resnext50_32x4d_1000': resnext50_32x4d,
    'ResNet10x2_1000': wide_resnet10_2,
    'ResNet18x2_1000': wide_resnet18_2,
    'ResNet34x2_1000': wide_resnet34_2,
    'wrn_50_2_1000': wide_resnet50_2,

    'MobileNetV2_Imagenet_1000': mobilenet_v2,
    'ShuffleV2_Imagenet_1000': shufflenet_v2_x1_0,
}
