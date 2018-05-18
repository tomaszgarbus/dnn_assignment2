from PIL import Image
import numpy as np
import os
import random

from constants import INPUT_SIZE, NUM_LABELS, VAL_SIZE, SYLVESTER
if not SYLVESTER:
    import matplotlib.pyplot as plt


class Loader:
    train_img_names = []
    val_img_names = []

    def __init__(self):
        self.train_img_names = sorted(list(os.walk('assignment2/training/images'))[0][2])
        self.train_img_names = list(map(lambda s: s[:-4], self.train_img_names))
        self.train_img_names = list(filter(lambda x: x[0] != '.', self.train_img_names))

        # Extract validation set.
        self.val_img_names = self.train_img_names[-VAL_SIZE:]
        self.train_img_names = self.train_img_names[:-VAL_SIZE]

    @staticmethod
    def load_img_by_name(name, dir='training'):
        if os.path.isfile('assignment2_resized/{0}/images/{1}.jpg'.format(dir, name)):
            img = Image.open('assignment2_resized/{0}/images/{1}.jpg'.format(dir, name))
        else:
            img = Image.open('assignment2/{0}/images/{1}.jpg'.format(dir, name))
        return np.array(img)

    @staticmethod
    def load_labels_by_name(name):
        if os.path.isfile('assignment2_resized/training/labels_plain/{0}.png'.format(name)):
            img = Image.open('assignment2_resized/training/labels_plain/{0}.png'.format(name))
        else:
            img = Image.open('assignment2/training/labels_plain/{0}.png'.format(name))
        return np.array(img)

    @staticmethod
    def resize_img(img):
        if img.shape[:2] != tuple(INPUT_SIZE):
            tmp = Image.fromarray(img, mode='RGB')
            tmp = tmp.resize(INPUT_SIZE)
            return np.array(tmp)
        else:
            return img

    @staticmethod
    def resize_labels(labels):
        if labels.shape[:2] != tuple(INPUT_SIZE):
            tmp = Image.fromarray(labels, mode='L')
            tmp = tmp.resize(INPUT_SIZE)
            return np.array(tmp)
        else:
            return labels

    def load_random_img_and_label(self):
        name = random.choice(self.train_img_names)
        return self.load_img_by_name(name), self.load_labels_by_name(name)

    def load_val_img_and_label(self, img_no):
        name = self.val_img_names[img_no]
        return self.load_img_by_name(name), self.load_labels_by_name(name)

    @staticmethod
    def show_image_or_labels(iol) -> None:
        if not SYLVESTER:
            plt.imshow(iol)
            plt.show()

    @staticmethod
    def labels_to_one_hot(labels):
        onehot = np.zeros(INPUT_SIZE + [NUM_LABELS])
        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1]):
                onehot[x][y][labels[x][y]] = 1
        return onehot

    @staticmethod
    def flip_img_or_labels(iol):
        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1] // 2):
                iol[x][y], iol[x][INPUT_SIZE[1]-1-y] =\
                    np.copy(iol[x][INPUT_SIZE[1]-1-y]), np.copy(iol[x][y])
        return iol

    def _preprocess_img_and_labels(self, img, labels, flip=None):
        img = self.resize_img(img)
        labels = self.resize_labels(labels)

        # Apply horizontal flip to half of images
        if flip or (flip is None and random.getrandbits(1)):
            img = self.flip_img_or_labels(img)
            labels = self.flip_img_or_labels(labels)

        img = img / 255
        labels = self.labels_to_one_hot(labels)
        return img, labels

    def prepare_batch(self, size):
        x = []
        y = []
        for img_no in range(size):
            img, labels = self.load_random_img_and_label()
            img, labels = self._preprocess_img_and_labels(img, labels)
            x.append(img)
            y.append(labels)
        x = np.array(x)
        y = np.array(y)
        return x, y

    def validation_batch(self, img_no_first, img_no_last, flip=None):
        x = []
        y = []
        for img_no in range(img_no_first, img_no_last + 1):
            img, labels = self.load_val_img_and_label(img_no)
            img, labels = self._preprocess_img_and_labels(img, labels, flip=flip)
            x.append(img)
            y.append(labels)
        x = np.array(x)
        y = np.array(y)
        return x, y


if __name__ == '__main__':
    loader = Loader()
    for i in range(5):
        pair = loader.load_random_img_and_label()
        pair = loader.resize_img(pair[0]), loader.resize_labels(pair[1])
        loader.show_image_or_labels(pair[0])
        loader.show_image_or_labels(pair[1])
        # print(l.load_random_img_and_label()[0].shape[0]/l.load_random_img_and_label()[0].shape[1])