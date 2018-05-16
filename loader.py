from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt


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

    def load_random_img_and_label(self):
        name = random.choice(self.img_names)
        return self.load_img_by_name(name), self.load_labels_by_name(name)


def show_image(img) -> None:
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    l = Loader()
    for i in range(100):
        print(l.load_random_img_and_label()[0].shape[0]/l.load_random_img_and_label()[0].shape[1])