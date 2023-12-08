import random


def RandomRotation(img):
    return img.rotate(90 * random.randint(0, 3))