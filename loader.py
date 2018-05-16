from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt

from constants import INPUT_SIZE, NUM_LABELS


class Loader:
    img_names = []

    def __init__(self):
        self.img_names = sorted(list(os.walk('assignment2/training/images'))[0][2])
        self.img_names = list(map(lambda s: s[:-4], self.img_names))
        pass

    @staticmethod
    def load_img_by_name(name, dir='training'):
        img = Image.open('assignment2/{0}/images/{1}.jpg'.format(dir, name))
        return np.array(img)

    @staticmethod
    def load_labels_by_name(name):
        img = Image.open('assignment2/training/labels_plain/{0}.png'.format(name))
        return np.array(img)

    @staticmethod
    def resize_img(img):
        tmp = Image.fromarray(img, mode='RGB')
        tmp = tmp.resize(INPUT_SIZE)
        return np.array(tmp)

    @staticmethod
    def resize_labels(labels):
        tmp = Image.fromarray(labels, mode='L')
        tmp = tmp.resize(INPUT_SIZE)
        return np.array(tmp)

    def load_random_img_and_label(self):
        name = random.choice(self.img_names)
        return self.load_img_by_name(name), self.load_labels_by_name(name)

    @staticmethod
    def show_image_or_labels(iol) -> None:
        plt.imshow(iol)
        plt.show()

    @staticmethod
    def labels_to_one_hot(labels):
        onehot = np.zeros(INPUT_SIZE + [NUM_LABELS])
        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1]):
                onehot[x][y][labels[x][y]] = 1
        return onehot

    def prepare_batch(self, size):
        x = []
        y = []
        for i in range(size):
            img, labels = self.load_random_img_and_label()
            img = self.resize_img(img)
            labels = self.resize_labels(labels)
            img = img / 255
            labels = self.labels_to_one_hot(labels)
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