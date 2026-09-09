"""Unmodified torchvision DeepLabV3 / ResNet50 public training API."""

from torchvision.models.segmentation import deeplabv3_resnet50
from unet_train import train


def model_factory(width):
    del width
    return deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=False, num_classes=3)


if __name__ == "__main__":
    train(model_factory)
