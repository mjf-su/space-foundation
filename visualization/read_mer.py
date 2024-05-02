import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from tqdm import tqdm
import pickle

ROOT = pathlib.Path().cwd()

# --- MER Directory ---
print("MER Directory")
image_dir = ROOT / "mer" / "images" / "eff"
label_dir = ROOT / "mer" / "labels" / "train" / "merged-unmasked"

# store unique image filenames
unique_image_names = set()
for labeled_image in label_dir.glob("*.png"):
    unique_image_names.add(labeled_image.stem.split("_")[0])

for image_name in tqdm(unique_image_names):
    reviews = [] # compile semantic labels from each reviewer
    for labeled_image_path in label_dir.glob(image_name + "*"):
        labeled_image = np.array(Image.open(labeled_image_path))
        soil, bedrock, sand, bigrock = np.zeros(labeled_image.shape), np.zeros(labeled_image.shape), np.zeros(labeled_image.shape), np.zeros(labeled_image.shape)

        soil[np.where(labeled_image == 0)] = 1
        bedrock[np.where(labeled_image == 1)] = 1
        sand[np.where(labeled_image == 2)] = 1
        bigrock[np.where(labeled_image == 3)] = 1    

        reviews.append(np.stack((soil, bedrock, sand, bigrock)))

    meta_reviews = np.sum(np.stack(reviews), axis = 0) # sum all binary mask from each review
    semantic_segmentation = np.argmax(meta_reviews, axis = 0) # record dominant class for each pixel
    soil, bedrock, sand, bigrock = np.split(meta_reviews, 4)
    soil, bedrock, sand, bigrock = soil.squeeze(0), bedrock.squeeze(0), sand.squeeze(0), bigrock.squeeze(0)

    soil[np.where((semantic_segmentation != 0) + (soil < 3))] = 0 # zero non-dominant mask or mask without at least three reviewers in agreement
    bedrock[np.where((semantic_segmentation != 1) + (bedrock < 3))] = 0
    sand[np.where((semantic_segmentation != 2) + (sand < 3))] = 0
    bigrock[np.where((semantic_segmentation != 3) + (bigrock < 3))] = 0

    soil[np.where(soil > 0)] = 1 # squash activations to 1
    bedrock[np.where(bedrock > 0)] = 1
    sand[np.where(sand > 0)] = 1
    bigrock[np.where(bigrock > 0)] = 1

    fig, axes = plt.subplots(2, 3)
    image = np.array(Image.open(image_dir / (image_name + ".JPG")).convert("RGB"))

    axes[0][0].imshow(image)
    axes[0][0].set_title("image")

    axes[0][1].imshow(soil)
    axes[0][1].set_title("soil")

    axes[0][2].imshow(bedrock)
    axes[0][2].set_title("bedrock")

    axes[1][0].imshow(sand)
    axes[1][0].set_title("sand")

    axes[1][1].imshow(bigrock)
    axes[1][1].set_title("big rock")
    plt.show()
