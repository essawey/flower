# # segmentation_models.py

# from .Segformer import Segformer
# import segmentation_models_pytorch as smp


# Unet_model = smp.Unet(
#     encoder_name="mit_b0",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=6,
#     activation='softmax2d' # None for logits & 'softmax2d' for multiclass segmentation
# )


# DeepLabV3Plus_model = smp.DeepLabV3Plus(
#     encoder_name='efficientnet-b4',
#     encoder_depth=5, #  in range [3, 5]
#     encoder_weights='imagenet',
#     in_channels=3,
#     classes=6,
#     activation='softmax2d' # None for logits & 'softmax2d' for multiclass segmentation
# )

# Segformer_model = Segformer(
#     encoder_name='efficientnet-b4',
#     encoder_depth=5, #  in range [3, 5]
#     encoder_weights='imagenet',
#     decoder_segmentation_channels=256,
#     in_channels=3,
#     classes=6,
#     activation='softmax2d' # None for logits & 'softmax2d' for multiclass segmentation
# )

