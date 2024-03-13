# import config
# import pyheif

import os
import shutil
from PIL import Image
import numpy as np
import time
from random import sample

from utils.data import resize_and_save


# import argparse
# import os


# def read_heic(file_path):
#     heif_file = pyheif.read(file_path)
#     return Image.frombytes(
#         heif_file.mode,
#         heif_file.size,
#         heif_file.data,
#         "raw",
#         heif_file.mode,
#         heif_file.stride,
#     )


# def get_parser():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--data_dir", default="data/SIGNS", help="Directory with the SIGNS dataset"
#     )
#     parser.add_argument(
#         "--output_dir", default="data/64x64_SIGNS", help="Where to write the new data"
#     )

#     return parser


# # Define the directories


# # Function to process images
# def process_images(directory, train_ratio=0.8):
#     base_dir = "./data/p_images"
#     train_dir = "./data/train/1"
#     test_dir = "./data/test/1"

#     # Create train and test directories if they don't exist
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)

#     # List all the subdirectories in the base directory
#     subdirectories = [
#         os.path.join(directory, d)
#         for d in os.listdir(directory)
#         if os.path.isdir(os.path.join(directory, d))
#     ]

#     # Initialize a list to hold all image paths
#     image_paths = []

#     # Loop through all subdirectories to get image paths
#     for subdirectory in subdirectories:
#         image_paths.extend(
#             [
#                 os.path.join(subdirectory, f)
#                 for f in os.listdir(subdirectory)
#                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".heic"))
#             ]
#         )

#     # Shuffle image paths to ensure random distribution
#     np.random.shuffle(image_paths)

#     # Split images into train and test sets based on the train_ratio
#     split_index = int(train_ratio * len(image_paths))
#     train_images = image_paths[:split_index]
#     test_images = image_paths[split_index:]

#     # Function to convert and resize image
#     def convert_and_resize(image_path, output_dir):
#         # Open the image
#         with Image.open(image_path) as img:
#             # Convert HEIC to JPG if needed
#             if image_path.lower().endswith(".heic"):
#                 img = read_heic(image_path)
#             else:
#                 img = Image.open(image_path)
#             # Resize image
#             img = img.resize((224, 224))
#             # Save to the output directory with a JPG extension
#             img.save(
#                 os.path.join(
#                     output_dir, os.path.basename(image_path).split(".")[0] + ".jpg"
#                 ),
#                 "JPEG",
#             )

#     # Process train images
#     for image_path in train_images:
#         convert_and_resize(image_path, train_dir)

#     # Process test images
#     for image_path in test_images:
#         convert_and_resize(image_path, test_dir)


# # Call the function to process images
# # process_images(base_dir)
# #

# if __name__ == "__main__":
#     # args = get_parser().parse_args()
#     # assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(
#     #     args.data_dir
#     # )
#     process_images("data/p_images")


def split_dataset():
    source_dir_0 = "./data/train_images/0"
    source_dir_1 = "./data/train_images/1"
    test_dir = "./data/test_images"
    test_ratio = 0.2

    # Create test directories
    os.makedirs(os.path.join(test_dir, "0"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "1"), exist_ok=True)

    def move_images(source, dest, ratio):
        files = [
            f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))
        ]

        # Calculate the number of files to move
        n_test_files = int(len(files) * ratio)

        # Randomly select files to move
        test_files = sample(files, n_test_files)

        # Move the files
        for f in test_files:
            shutil.move(os.path.join(source, f), os.path.join(dest, f))

    move_images(source_dir_0, os.path.join(test_dir, "0"), test_ratio)
    move_images(source_dir_1, os.path.join(test_dir, "1"), test_ratio)


if __name__ == "__main__":
    # data_directory = "./data/p_images"
    # out_directory = "./data/train_images/1"
    # images = []
    # count = 1
    # for directory in [f for f in os.listdir(data_directory) if not f.startswith(".")]:
    #     print(directory)
    #     for image in os.listdir(os.path.join(data_directory, directory)):
    #         resize_and_save(
    #             os.path.join(data_directory, directory, image), out_directory, count
    #         )
    #         count += 1
    split_dataset()
