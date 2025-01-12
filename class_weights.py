import os
import glob
import numpy as np

def class_weights():
    mask_dir = os.path.join(os.getcwd(), 'PanNuke', 'data', 'Patched', 'Train', '**', '*.npy')
    mask_file_list = glob.glob(mask_dir, recursive=True)
    mask_file_list = sorted(mask_file_list)

    class_pixel_counts = {}  # Dictionary to accumulate pixel counts across all files

    for path in mask_file_list:
        # Load the numpy array
        array = np.load(path)
        
        # Get unique class labels and their pixel counts
        unique_classes, counts = np.unique(array, return_counts=True)
        
        # Update the global class pixel count dictionary
        for cls, count in zip(unique_classes, counts):
            if cls in class_pixel_counts:
                class_pixel_counts[cls] += count
            else:
                class_pixel_counts[cls] = count

    # Calculate class weights based on the total pixel count
    total_pixels = sum(class_pixel_counts.values())
    class_weights = {cls: total_pixels / count for cls, count in class_pixel_counts.items()}

    return class_weights

# Call the function and print the computed class weights
weights = compute_class_weights()
print(weights)
