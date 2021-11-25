from tensorflow import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    'Generate data for keras'
    def __init__(self, image_paths, batch_size, augment, shuffle, normalize=False, 
                minimum=0, maximum=0, lstm=False):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.normalize = normalize
        self.minimum = minimum
        self.maximum = maximum
        self.lstm = lstm
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_paths) / self.batch_size))
  

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
              np.random.shuffle(self.indices)
  
    def __getitem__(self, index):
        'Generate one batch of data'
        # select indices of data for next batch
        indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]

        # select data and load images
        imgs_lbls = []
        for i in indices:
            img_lbl = np.load(self.image_paths[i])
            # img_lbl = np.moveaxis(img_lbl, 0, -1)
            imgs_lbls.append(img_lbl)

        # preprocess and augment data
        if self.augment:
            img_lbls = self.augmentor(imgs_lbls)
    
        images = []
        labels = []
        for img_lbl in imgs_lbls:
            image = img_lbl[:12, :, :]
            if not self.lstm:
                image = np.moveaxis(image, 0, -1)
            images.append(image)
            label = img_lbl[12:, :, :]
            label = np.moveaxis(label, 0, -1)
            labels.append(label)
        images = np.array(images, dtype=np.float32)
        if self.normalize:
            images = (images - self.minimum) / (self.maximum - self.minimum)
        return images, np.array(labels, dtype=np.float32)
  
    def augmentor(self, imgs_lbls):
        k = np.random.randint(0, 4)
        imgs_lbls = [np.rot90(img_lbl, axes=(0, 1), k=k) for img_lbl in imgs_lbls]

        if np.random.randint(0, 2) == 0:
            imgs_lbls = [np.fliplr(img_lbl) for img_lbl in imgs_lbls]
        if np.random.randint(0, 2) == 0:
            imgs_lbls = [np.flipud(img_lbl) for img_lbl in imgs_lbls]
    

        return imgs_lbls