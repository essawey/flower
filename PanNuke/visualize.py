import matplotlib.pyplot as plt
import torch
import numpy as np

def binarizeChannel(masks):
    masksNum, channels, _, _ = masks.shape
    masks_binary = np.empty_like(masks)
    # Loop through each mask and channel
    for mask_idx in range(masksNum):
        for channel_idx in range(channels):
            current_channel = masks[mask_idx, channel_idx]
            # set higher values than 1 to 1
            current_channel[current_channel > 1] = 1
            masks_binary[mask_idx, channel_idx] = current_channel
    return masks_binary

def show_image(image):
    means = [0.7039875984191895, 0.5724194049835205, 0.7407296895980835]
    stds = [0.12305392324924469, 0.16210812330245972, 0.14659656584262848]
    means_tensor = torch.tensor(means).view(3, 1, 1)  # Shape: [3, 1, 1]
    stds_tensor = torch.tensor(stds).view(3, 1, 1)    # Shape: [3, 1, 1]
    # Reverse normalization: (image * std) + mean
    image = image * stds_tensor + means_tensor
    image = image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def show_mask(mask):
    masksLabels = [
        'Neoplastic cells',
        'Inflammatory',
        'Connective/Soft tissue cells',
        'Dead Cells',
        'Epithelial',
        'Background',
    ]

    # Ensure mask is a numpy array
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    # Ensure mask shape is (6, height, width)
    if mask.shape[0] != 6:
        if 1 in mask.shape:
            mask = np.squeeze(mask)
        mask = np.eye(6)[mask]  # One-hot encode to (height, width, 6)
        mask = np.moveaxis(mask, -1, 0)  # Rearrange to (6, height, width)

    labels_idx = {label_idx: label for label_idx, label in enumerate(masksLabels)}

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    channel_images = []

    for channel_index in range(mask.shape[0]):
        max_value = np.max(mask[channel_index, :, :])
        colors = plt.cm.get_cmap('tab20', int(max_value + 1))

        row = channel_index // 3
        col = channel_index % 3

        ax = axes[row, col]
        im = ax.imshow(mask[channel_index, :, :], cmap=colors, vmin=0, vmax=max_value)
        ax.set_title(f'Channel {channel_index} : {labels_idx[channel_index]}')
        ax.axis('off')
        channel_images.append(im)

    cbar_ax = fig.add_axes((0.15, -0.02, 0.7, 0.03))
    fig.colorbar(channel_images[-1], cax=cbar_ax, orientation='horizontal')
    plt.tight_layout()
    plt.show()


