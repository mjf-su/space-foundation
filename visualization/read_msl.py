from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import numpy as np
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import pickle

import cv2

ROOT = pathlib.Path().cwd()

with open("msl_datalog.pickle", "rb") as f_id:
    data = pickle.load(f_id)

for image_name, image_data in data.items():
    soil_present = np.sum(image_data["patch-population"][0]) > 50*50
    bedrock_present = np.sum(image_data["patch-population"][1]) > 50*50
    sand_present = np.sum(image_data["patch-population"][2]) > 50*50
    bigrock_present = np.sum(image_data["patch-population"][3]) > 50*50
    
    if np.sum([soil_present, bedrock_present, sand_present, bigrock_present]) > 1:
        image_path = ROOT / "msl" / "images" / "edr" / (image_name + ".JPG")
        label_path = ROOT / "msl" / "labels" / "train" / (image_name + ".png")

        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))

        soil = np.zeros(image.shape)
        bedrock = np.zeros(image.shape)
        sand = np.zeros(image.shape)
        bigrock = np.zeros(image.shape)

        soil[np.where(label == 0)] = 1 # soil activation
        bedrock[np.where(label == 1)] = 1 # bedrock activation
        sand[np.where(label == 2)] = 1 # sand activation
        bigrock[np.where(label == 3)] = 1 # big rock activation

        contours, heiarchy = cv2.findContours(bedrock, )

        fig, axes = plt.subplots(2, 3)
        axes[0][0].imshow(image)
        axes[0][1].imshow(label); axes[0][1].set_title("label")
        axes[0][2].imshow(soil); axes[0][2].set_title("soil")
        axes[1][0].imshow(bedrock); axes[1][0].set_title("bedrock")
        axes[1][1].imshow(sand); axes[1][1].set_title("sand")
        axes[1][2].imshow(bigrock); axes[1][2].set_title("big rock")
        plt.show()


