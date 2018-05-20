from PIL import Image, ImageOps
import numpy as np
import os
import random
from typing import Optional

from constants import INPUT_SIZE, NUM_LABELS, VAL_SIZE, MAX_CROP_MARGIN, SYLVESTER
if not SYLVESTER:
    import matplotlib.pyplot as plt


class Loader:
    train_img_names = []
    val_img_names = []
    train_img_cache = {}
    train_labels_cache = {}

    def __init__(self):
        self.train_img_names = sorted(list(os.walk('assignment2/training/images'))[0][2])
        self.train_img_names = list(map(lambda s: s[:-4], self.train_img_names))
        self.train_img_names = list(filter(lambda x: x[0] != '.', self.train_img_names))

        # Extract validation set.
        self.val_img_names = self.train_img_names[-VAL_SIZE:]
        self.train_img_names = self.train_img_names[:-VAL_SIZE]

    def load_img_by_name(self, name, directory='training', force_full_size=True):
        if not force_full_size and name in self.train_img_cache:
            img = self.train_img_cache[name]
        else:
            img = Image.open('assignment2/{0}/images/{1}.jpg'.format(directory, name))
            if not force_full_size:
                img = img.resize(INPUT_SIZE)
                self.train_img_cache[name] = img
        return np.array(img)

    def load_labels_by_name(self, name, force_full_size=True):
        if not force_full_size and name in self.train_labels_cache:
            img = self.train_labels_cache[name]
        else:
            img = Image.open('assignment2/training/labels_plain/{0}.png'.format(name))
            if not force_full_size:
                img = img.resize(INPUT_SIZE)
                self.train_labels_cache[name] = img
        return np.array(img)

    @staticmethod
    def resize_img(img, size=INPUT_SIZE):
        tmp = Image.fromarray(img, mode='RGB')
        tmp = tmp.resize(size, resample=Image.NEAREST)
        return np.array(tmp)

    @staticmethod
    def resize_labels(labels, size=INPUT_SIZE):
        tmp = Image.fromarray(labels, mode='L')
        tmp = tmp.resize(size, resample=Image.NEAREST)
        return np.array(tmp)

    def load_random_img_and_label(self, force_full_size=True):
        name = random.choice(self.train_img_names)
        return self.load_img_by_name(name, force_full_size=force_full_size),\
            self.load_labels_by_name(name, force_full_size=force_full_size),\
            name

    def load_val_img_and_label(self, img_no, force_full_size=True):
        name = self.val_img_names[img_no]
        return self.load_img_by_name(name, force_full_size=force_full_size),\
            self.load_labels_by_name(name, force_full_size=force_full_size)

    @staticmethod
    def show_image_or_labels(iol: np.array) -> None:
        if not SYLVESTER:
            plt.imshow(iol)
            plt.show()

    @staticmethod
    def labels_to_one_hot(labels: np.array):
        onehot = np.zeros(INPUT_SIZE + [NUM_LABELS])
        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1]):
                onehot[x][y][labels[x][y]] = 1
        return onehot

    @staticmethod
    def transform_flip(img: Image, labels: Image):
        img = ImageOps.mirror(img)
        labels = ImageOps.mirror(labels)
        return img, labels

    @staticmethod
    def transform_crop(img: Image,
                       labels: Image,
                       margin_x: Optional[float] = None,
                       margin_y: Optional[float] = None):
        if margin_x is None:
            margin_x = np.random.rand() * MAX_CROP_MARGIN
        if margin_y is None:
            margin_y = np.random.rand() * MAX_CROP_MARGIN
        margin_x *= img.size[0]
        margin_y *= img.size[1]
        img = img.crop((margin_x, margin_y,
                        img.size[0] - margin_x, img.size[1] - margin_y))
        labels = labels.crop((margin_x, margin_y,
                              labels.size[0] - margin_x, labels.size[1] - margin_y))
        return img, labels

    @staticmethod
    def transform_rotate(img: Image, labels: Image, deg: Optional[float] = None):
        if deg is None:
            deg = np.random.rand() * 15.
        img = img.rotate(deg)
        labels = labels.rotate(deg)
        # TODO: smarter crop then margin = angle...
        img, labels = Loader.transform_crop(img, labels, margin_x=deg / 100, margin_y=deg / 100)
        return img, labels

    def _transform_img_and_labels(self, img, labels,
                                  flip: Optional[bool] = None,
                                  crop: Optional[bool] = None,
                                  rotate: Optional[bool] = None):
        # Convert to PIL.Image
        img = Image.fromarray(img, mode='RGB')
        labels = Image.fromarray(labels, mode='L')
        # Apply transformations
        if rotate or (rotate is None and random.getrandbits(1)):
            img, labels = self.transform_rotate(img, labels)
        if flip or (flip is None and random.getrandbits(1)):
            img, labels = self.transform_flip(img, labels)
        if crop or (crop is None and random.getrandbits(1)):
            img, labels = self.transform_crop(img, labels)
        # Convert back.
        img = np.array(img)
        labels = np.array(labels)
        return img, labels

    def _resize_and_convert_img_and_labels(self, img, labels):
        # Resize
        img = self.resize_img(img)
        labels = self.resize_labels(labels)
        # Convert to input format of the network.
        img = img / 255
        labels = self.labels_to_one_hot(labels)
        return img, labels

    def prepare_batch(self, size):
        x = []
        y = []
        for img_no in range(size):
            img, labels, name = self.load_random_img_and_label()
            img, labels = self._transform_img_and_labels(img, labels)
            img, labels = self._resize_and_convert_img_and_labels(img, labels)
            x.append(img)
            y.append(labels)
        x = np.array(x)
        y = np.array(y)
        return x, y

    def validation_batch(self, img_no_first, img_no_last, flip=None):
        x = []
        y = []
        orig_size = []
        for img_no in range(img_no_first, img_no_last + 1):
            img, orig_labels = self.load_val_img_and_label(img_no)
            img, orig_labels = self._transform_img_and_labels(img, orig_labels, flip=flip)
            img, labels = self._resize_and_convert_img_and_labels(img, orig_labels)

            x.append(img)
            y.append(labels)
            orig_size.append(orig_labels)
        x = np.array(x)
        y = np.array(y)
        return x, y, orig_size


if __name__ == '__main__':
    loader = Loader()
    for i in range(5):
        pair = loader.load_random_img_and_label()
        pair = loader.resize_img(pair[0]), loader.resize_labels(pair[1])
        loader.show_image_or_labels(pair[0])
        loader.show_image_or_labels(pair[1])
        # print(l.load_random_img_and_label()[0].shape[0]/l.load_random_img_and_label()[0].shape[1])
