import random


def random_crop(img, gt, size):
    # crop a [size*size] region both in img and gt
    width, height = img.size
    max_x = width - size
    max_y = height - size
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    region = (x, y, x + size, y + size)
    img = img.crop(region)
    gt = gt.crop(region)
    return img, gt


def resize_to_multiple_of_k(image, k):
    width, height = image.size
    new_height = int(height // k * k)
    new_width = int(width // k * k)
    image_resize = image.resize((new_width, new_height))
    return image_resize


def crop_to_multiple_of_k(image, k):
    width, height = image.size
    new_height = int(height // k * k)
    new_width = int(width // k * k)
    image_crop = image.crop((0, 0, new_width, new_height))
    return image_crop
