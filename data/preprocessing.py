import pickle
import rasterio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib as mpl
from copy import deepcopy
import matplotlib.patches as mpatches
import os
from tensorflow.keras.utils import to_categorical


# all labels
all_crops_key_values = {1: 'Corn', 2: 'Cotton', 3: 'Rice', 4: 'Sorghum', 5: 'Soybeans',
                    6: 'Sunflower', 10: 'Peanuts', 11: 'Tobacco', 12: 'Sweet Corn',
                    13: 'Pop or Orn Corn', 14: 'Mint', 21: 'Barley', 22: 'Durum Wheat',
                    23: 'Spring Wheat', 24: 'Winter Wheat', 25: 'Other Small Grains',
                    26: 'Dbl Crop WinWht/Soybeans', 27: 'Rye', 28: 'Oats', 29: 'Millet',
                    30: 'Spelts', 31: 'Canola', 32: 'Flaxseed', 33: 'Safflower',
                    34: 'Rape Seed', 35: 'Mustard', 36: 'Alfalfa', 37: 'Other Hay/Non Alfalfa',
                    38: 'Camelina', 39: 'Buckwheat', 41: 'Sugarbeets', 42: 'Dry Beans', 
                    43: 'Potatoes', 44: 'Other Crops', 45: 'Sugarcane', 46: 'Sweet Potatoes',
                    47: 'Misc Vegs & Fruits', 48: 'Watermelons', 49: 'Onions', 50: 'Cucumbers',
                    51: 'Chick Peas', 52: 'Lentils', 53: 'Peas', 54: 'Tomatoes', 
                    55: 'Caneberries', 56: 'Hops', 57: 'Herbs', 58: 'Clover/Wildflowers',
                    59: 'Sod/Grass Seed', 60: 'Switchgrass', 61: 'Fallow/Idle Cropland',
                    63: 'Forest', 64: 'Shrubland', 65: 'Barren', 66: 'Cherries', 
                    67: 'Peaches', 68: 'Apples', 69: 'Grapes', 70: 'Christmas Trees',
                    71: 'Other Tree Crops', 72: 'Citrus', 74: 'Pecans', 75: 'Almonds',
                    76: 'Walnuts', 77: 'Pears', 81: 'Clouds/No Data', 82: 'Developed', 
                    83: 'Water', 87: 'Wetlands', 88: 'Nonag/Undefined', 92: 'Aquaculture',
                    111: 'Open Water', 112: 'Perennial Ice/Snow', 121: 'Developed/Open Space',
                    122: 'Developed/Lpw Intensity', 123: 'Developed/Med Intensity',
                    124: 'Developed/High Intensity', 131: 'Barren', 141: 'Deciduous Forest',
                    142: 'Evergreen Forest', 143: 'Mixed Forest', 152: 'Shrubland',
                    176: 'Grassland/Pasture', 190: 'Woody Wetlands', 195: 'Herbaceous Wetlands',
                    204: 'Pistachios', 205: 'Triticale', 206: 'Carrots', 207: 'Asparagus', 208: 'Garlic',
                    209: 'Cantaloupes', 210: 'Prunes', 211: 'Olives', 212: 'Oranges',
                    213: 'Honeydew Melons', 214: 'Broccoli', 215: 'Avocados', 216: 'Peppers',
                    217: 'Pomegranates',
                    218: 'Nectarines', 219: 'Greens', 220: 'Plums', 221: 'Strawberries',
                    222: 'Squash', 223: 'Apricots', 224: 'Vetch', 225: 'Dbl Crop WinWht/Corn',
                    226: 'Dbl Crop Oats/Corn', 227: 'Lettuce', 228: 'Dlb Crop Triticale/Corn',
                    229: 'Pumplins', 230: 'Dbl Crop Lettuce/Durum Wht', 231: 'Dbl Crop Lettuce/Cantaloupe',
                    232: 'Dbl Crop Lettuce/Cotton', 233: 'Dbl Crop Lettuce/Barley',
                    234: 'Dbl Crop Durum Wht/Sorghum', 235: 'Dbl Crop Barley/Sorghum', 236: 'Dbl Crop WinWht/Sorghum',
                    237: 'Dbl Crop Barley/Corn', 238: 'Dbl Crop WinWht/Cotton', 239: 'Dbl Crop Soybeans/Cotton',
                    240: 'Dbl Crop Soybeans/Oats', 241: 'Dbl Crop Corn/Soybeans', 242: 'Blueberries',
                    243: 'Cabbage', 244: 'Cauliflower', 245: 'Celery', 246: 'Radishes',
                    247: 'Turnips', 248: 'Eggplants', 249: 'Gourds', 250: 'Cranberries',
                    254: 'Dbl Crop Barley/Soybeans'}


# read raster file
def read_raster(path):
    # file format: .tif
    return rasterio.open('path', 'r').read().squeeze()


# plot label
def plot_label(label):
    plt.imshow(np.squeeze(label.read), cmap='hsv')
    plt.xticks([]); plt.yticks([])
    plt.show()


# plot sar image channels
def plot_sar_image_channels(image):
    dataset = image
    for i in range(12):
        data = dataset[i, :, :]
        plt.figure(i, figsize=(20, 10))
        plt.xticks([]); plt.yticks([])
        plt.imshow(data, cmap='gray')
    plt.show()


# extract unique labels
def extract_unique_labels(*labels):
    labels = [np.unique[label] for label in labels]
    labels = np.unique(labels)
    return labels


# crops of dataset
def dataset_crops(unique_labels):
    crops = list()

    for crop in unique_labels:
        crops.append(all_crops_key_values[crop])
    crops = np.array(crops)

    return crops


# extract crop indices
def crop_indices(labels, crops, unique_labels):
    crop_indices = {}
    for crop in crops:
        crp = np.where(crops == crop)
        indx = unique_labels[crp]
        print(f"{crop}: {crp}, {indx}")
        indices = np.where(labels == indx)
        crop_indices[crop] = indices
    return crop_indices


# count crops
def count_crops(crops, crop_indices):
    crop_counts = {}
    for crop in crops:
        crop_counts[crop] = crop_indices[crop][0].shape[0]

    return crop_counts



# sort crops
def sort_crops(crop_counts):
    sorted_crop_counts = sorted(crop_counts.items(), key=lambda x: x[1], reverse=True)  
    return sorted_crop_counts


# reverse key values
def rever_key_values(all_crops_key_values):
    reversed_all_crops_key_values = {value : key for (key, value) in all_crops_key_values.items()}
    return reversed_all_crops_key_values


# change labels coding in range 0-num classes
def preprocess_labels(corrected_labels, unique_labels, selected_crops):
    counter = 1
    for crop in unique_labels:
        if crop in selected_crops:
            # print('selected', all_crops_key_values[crop], crop)
            indx = np.where(corrected_labels == crop)
            corrected_labels[indx] = counter
            counter += 1
        else:
            # print('not selected', all_crops_key_values[crop], crop)
            indx = np.where(corrected_labels == crop)
            corrected_labels[indx] = 0
    
    return corrected_labels


# plot corrected labels
def plot_corrected_labels(labels, plt_labels):
    # plt_labels = ['other', 'Corn', 'Cotton', 'Rice']
    values = np.unique(labels.ravel())
    im = plt.imshow(labels, interpolation='none', cmap="gist_ncar")
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=plt_labels[i]) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.show()
    

# one-hot encoding
def one_hot_encoding(labels, num_classes):
    return to_categorical(labels, num_classes=num_classes)


# create pathches
def create_patches(data, label, path, name, frm, to):
    if not os.path.exists():
        os.path.makedirs(path)


    label = np.moveaxis(label[0], -1, 0)
    print(data.shape, label[..., frm:to].shape)
    all_dataset = np.concatenate((data, label[..., frm:to]), axis=0)
    print(all_dataset.shape)
    BATCH_SIZE = 256
    counter = 0
    for i in range(all_dataset.shape[1]//BATCH_SIZE):
        for j in range(all_dataset.shape[2]//BATCH_SIZE):
            sub_image = all_dataset[:, i*BATCH_SIZE:(i+1)*BATCH_SIZE, j*BATCH_SIZE:(j+1)*BATCH_SIZE]
#             print(sub_image.shape)
            counter += 1
#             print(counter)
            pattern0 = "{}/{}_{}.npy".format(path, name, counter) 
            with open(pattern0, 'wb') as f:
                # print(sub_image.shape)
                np.save(f, sub_image)