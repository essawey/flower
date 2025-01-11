

from .Segformer import Segformer

Segformer_model = Segformer(
    encoder_name='efficientnet-b4',
    encoder_depth=5, #  in range [3, 5]
    encoder_weights='imagenet',
    decoder_segmentation_channels=256,
    in_channels=3,
    classes=6,
    activation='softmax2d' # None for logits & 'softmax2d' for multiclass segmentation
)

import segmentation_models_pytorch as smp

# Segmentation Models encoder parameters
encoders = {
    # MobileNet
    0.43:  "timm-mobilenetv3_small_minimal_100",

    # MiT
    3: "mit_b0",
    13: "mit_b1",
    24: "mit_b2",
    44: "mit_b3",
    60: "mit_b4",
    81: "mit_b5"
}



Unet_model = smp.Unet(
    encoder_name="mit_b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=6,
    activation='softmax2d' # None for logits & 'softmax2d' for multiclass segmentation
)


DeepLabV3Plus_model = smp.DeepLabV3Plus(
    encoder_name='efficientnet-b4',
    encoder_depth=5, #  in range [3, 5]
    encoder_weights='imagenet',
    in_channels=3,
    classes=6,
    activation='softmax2d' # None for logits & 'softmax2d' for multiclass segmentation
)