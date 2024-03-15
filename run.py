import argparse
import os
from tqdm import trange

from ultralytics import YOLO
import torch
from torchvision import transforms
from torchvision.io import read_image
import numpy as np


from model.vit import vit
from utils.misc import load_checkpoint
import config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--classifier_model_weight",
        default="experiments/300_epochs/best.pth.tar",
        help="Load Classifier model weight",
    )

    parser.add_argument(
        "-d",
        "--data_dir",
        default="./evaluation/eval_folder_",
        help="Input images to predict",
    )

    parser.add_argument(
        "-y", "--yolo_weight", default="./yolo_weight/last_v5.pt", help="Yolo Weight"
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="./evaluation",
        help="Optional, name of the file in --model_dir containing weights to reload before training",
    )

    return parser


def load_images(image_paths):
    transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_names = []
    image_batch = []
    for image_path in image_paths:
        image = read_image(image_path)

        image_batch.append(transform(image))
        image_names.append([image_path.split("/")[-1]])

    return image_batch, image_names


def tensor_to_str(t):
    """
    t:
        x_1: left top
        y_1: left top
        x_2: right bottom
        y_2: right bottom
    """
    t = t.int()

    x = t[0]
    y = t[1]
    w = t[2] - t[0]
    h = t[3] - t[1]

    return f"{x}, {y}, {w}, {h}"


def main():
    args = get_parser().parse_args()

    # Load model
    print("initializing models")
    model = vit()
    load_checkpoint(args.classifier_model_weight, model)
    yolo = YOLO(args.yolo_weight)
    print("- done\n")

    # Load images
    print("loading images ...")
    image_paths = [
        os.path.join(args.data_dir, image_name)
        for image_name in os.listdir(args.data_dir)
    ]

    print(f"there are {len(image_paths)} to process")
    for i in trange(0, len(image_paths), 32):
        image_batch, image_names = load_images(image_paths[i : i + 32])
        image_batch = torch.stack(image_batch)

        model.eval()
        outputs = model(image_batch)
        outputs = outputs.detach().cpu().numpy()

        outputs = np.squeeze(outputs)
        # pred = (outputs > 0.5).astype(int)
        detect_output = yolo(
            # [os.path.join(args.data_dir, image_name) for image_name in image_names],
            image_paths[i : i + 10],
            save=True,
            conf=0.15,
            iou=0.3,
            verbose=False,
        )

        with open(os.path.join(args.output_dir, config.CLASSIFIER_OUTPUT), "a") as f:
            for img_name, output in zip(image_names, outputs):
                f.write(f"{img_name[0]}, {output:.2f} \n")

        with open(os.path.join(args.output_dir, config.DETECT_OUTPUT), "a") as f:
            for result in detect_output:
                image_name = result.path.split("/")[-1]

                if result.boxes:
                    boxes = result.boxes
                    for box in boxes:
                        f.write(f"{image_name}, {tensor_to_str(box.xyxy[0])} \n")

        print("- done. \n")


if __name__ == "__main__":
    main()
