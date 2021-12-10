import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils import *
import argparse
import numpy as np


def create_mask(pred_mask):
    """convert predection to binary img
     if pixel value>0.5 white else black

    Args:
        pred_mask: model predection

    Returns:
         pred_mask:  binary image-lane mask
    """
    pred_mask[pred_mask > 0.5] = 255
    pred_mask[pred_mask < 0.5] = 0

    return pred_mask


def preprocess_img(image_path, size=512):
    """preprocess image for feeding into model

    Args:
        image_path: path to input img
        size (int, optional): img_scale. Defaults to 512.

    Returns:
        img_tensor: (1,size,size,3) img tensor
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    input_image = tf.image.resize(image, (size, size), method="nearest")
    input_image = input_image / 255.0
    input_image = tf.expand_dims(input_image, axis=0)
    return input_image


def get_inference(model, input_img):
    """run model inference

    Args:
        model :unet model
        input_img :input img tensor

    Returns:
       out_img : predecited binary image (lane mask)
    """

    pred = model.predict(input_img)
    out_img = create_mask(pred[0])
    return out_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run inference")

    parser.add_argument("--source", type=str, help="path to image or folder")
    parser.add_argument(
        "--output",
        type=str,
        help="path to results folder",
        default="./results",
    )
    parser.add_argument("--weights", type=str, help="weights path")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    model = load_model(
        args.weights,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_loss": dice_loss,
            "dice_coeff": dice_coeff,
        },
    )
    if os.path.isfile(args.source):
        print(args.source)
        img_path = args.source
        input_image = preprocess_img(img_path)
        predection = get_inference(model, input_image)
        img = predection
        img_name = "mask_" + img_path.split("/")[-1]
        cv2.imwrite("./results/" + img_name, img)

    elif args.source:
        for (root, dirs, files) in os.walk(args.source):
            for filen in files:
                img_path = os.path.join(root, filen)
                print("processing", img_path)
                input_img = preprocess_img(img_path)
                predection = get_inference(model, input_img)
                img_name = "mask_" + img_path.split("/")[-1]
                out_img_path = os.path.join(args.output, img_name)
                cv2.imwrite(out_img_path + img_name, predection)
