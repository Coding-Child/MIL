import random
from PIL import ImageDraw


class RandomErasing:
    def __init__(self, probability=0.5, area_ratio_range=(0.02, 0.33), aspect_ratio_range=(0.3, 3.3), fill_color=(0, 0, 0)):
        self.probability = probability
        self.area_ratio_range = area_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.fill_color = fill_color

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        w, h = img.size
        area = w * h

        target_area = random.uniform(*self.area_ratio_range) * area
        aspect_ratio = random.uniform(*self.aspect_ratio_range)

        h_erase = int(round((target_area * aspect_ratio) ** 0.5))
        w_erase = int(round((target_area / aspect_ratio) ** 0.5))

        if w_erase > w or h_erase > h:
            return img

        x = random.randint(0, w - w_erase)
        y = random.randint(0, h - h_erase)

        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + w_erase, y + h_erase], fill=self.fill_color)

        return img


def RandomRotation(img):
    return img.rotate(90 * random.randint(0, 3))