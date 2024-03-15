import os
import shutil
from PIL import Image
import numpy as np
import time
from random import sample

from utils.data import resize_and_save


if __name__ == "__main__":
    data_directory = "./data/is"
    out_directory = "./data/out"
    images = []
    count = 2200

    for image in os.listdir(data_directory):
        if image.lower().split(".")[-1] in ["jpg", "jpeg", "png"]:
            resize_and_save(os.path.join(data_directory, image), out_directory, count)

            count += 1

    # resize_and_save("data/example_images/sample3.jpg", out_directory, 1090)
    # for directory in [f for f in os.listdir(data_directory) if not f.startswith(".")]:
    #     print(directory)
    # for image in os.listdir(os.path.join(data_directory, directory)):
    #     resize_and_save(
    #         os.path.join(data_directory, directory, image), out_directory, count
    #     )
    #     count += 1
    # split_dataset()
