# Processing

To make patches from main dataset.

## Usage

```python
# All labels of CDL dataset
all_crops_key_values

# to read raster files 
read_raster(path)

# to plot labels 
plot_label(label)

# To plot channels of each SAR image. Each sar image is created
# by concatenating of 12 months images
plot_sar_image_channels(image)

# to extract unique labels, to find out the name of labels
extract_unique_labels(*labels)

# to find names of crops
dataset_crops(unique_labels):

# to count crops
count_crops(crops, crop_indices)

# to sort crops based on their counts
sort_crops(crop_counts)

# to change labels coding in range 0-num classes
preprocess_labels(corrected_labels, unique_labels, selected_crops)

# to plot corrected labels
plot_corrected_labels(labels, plt_labels)

# to one-hot encoding
one_hot_encoding(labels, num_classes)

# to create pathches
create_patches(data, label, path, name, frm, to)
```
