import os

import cv2

from src.colorization.Zhang import create_colorized_pictures


def colorize_zhang_siggraph17(image_path, image_name):
    create_colorized_pictures(model='siggraph17', img_path=image_path, save_prefix=image_name)
    return f'../../resources/additional/{image_name}_siggraph17.png'


# for all picture in the folder, colorize them and save them in the same folder
for image in os.listdir('../../resources/additional/'):
    image_path = f'../../resources/additional/{image}'
    image_name = image.split('.')[0]
    colorize_zhang_siggraph17(image_path, image_name)