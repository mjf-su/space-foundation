import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from tqdm import tqdm
import pickle

ROOT = pathlib.Path().cwd()

msl_datalog = {}
mer_datalog = {}
visualize = False

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
        labeled_image = np.array(Image.open(labeled_image_path).convert("RGB"))[:, :, 0]
        soil, bedrock, sand, bigrock = np.zeros(labeled_image.shape), np.zeros(labeled_image.shape), np.zeros(labeled_image.shape), np.zeros(labeled_image.shape)

        soil[np.where(labeled_image == 0)] = 1
        bedrock[np.where(labeled_image == 1)] = 1
        sand[np.where(labeled_image == 2)] = 1
        bigrock[np.where(labeled_image == 3)] = 1    

        reviews.append(np.stack((soil, bedrock, sand, bigrock)))
    mer_datalog[image_name] = {}
    mer_datalog[image_name]["patch-population"] = np.zeros((4, 3, 3))
    mer_datalog[image_name]["object-centriods"] = np.zeros((4, 2))

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

    soil_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(soil == 1)).T) if np.concatenate(np.where(soil == 1)).size else -1
    bedrock_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(bedrock == 1)).T) if np.concatenate(np.where(bedrock == 1)).size else -1
    sand_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(sand == 1)).T) if np.concatenate(np.where(sand == 1)).size else -1
    bigrock_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(bigrock == 1)).T) if np.concatenate(np.where(bigrock == 1)).size else -1

    soil_center = np.flip(soil_mean.cluster_centers_[0], axis = -1) if soil_mean != -1 else np.array([-1, -1])
    bedrock_center = np.flip(bedrock_mean.cluster_centers_[0], axis = -1) if bedrock_mean != -1 else np.array([-1, -1])
    sand_center = np.flip(sand_mean.cluster_centers_[0], axis = -1) if sand_mean != -1 else np.array([-1, -1])
    bigrock_center = np.flip(bigrock_mean.cluster_centers_[0], axis = -1) if bigrock_mean != -1 else np.array([-1, -1])
    mer_datalog[image_name]["object-centriods"] = np.vstack((soil_center, bedrock_center, sand_center, bigrock_center)) # row 0 -> soil, 1 -> bedrock ... 

    if visualize:
        fig, axes = plt.subplots(2, 3)
        image = np.array(Image.open(image_dir / (image_name + ".JPG")).convert("RGB"))

        axes[0][0].imshow(image)
        axes[0][0].set_title("image")

        axes[0][1].imshow(soil)
        axes[0][1].set_title("soil")
        axes[0][1].scatter(soil_center[0], soil_center[1], marker = '*', c = 'k')

        axes[0][2].imshow(bedrock)
        axes[0][2].set_title("bedrock")
        axes[0][2].scatter(bedrock_center[0], bedrock_center[1], marker = '*', c = 'k')

        axes[1][0].imshow(sand)
        axes[1][0].set_title("sand")
        axes[1][0].scatter(sand_center[0], sand_center[1], marker = '*', c = 'k')

        axes[1][1].imshow(bigrock)
        axes[1][1].set_title("big rock")
        axes[1][1].scatter(bigrock_center[0], bigrock_center[1], marker = '*', c = 'k')
        plt.show()
    
    assert semantic_segmentation.shape[0] == semantic_segmentation.shape[1]
    increment = semantic_segmentation.shape[0] // 3
    for i, object in enumerate([soil, bedrock, sand, bigrock]):
        for j in range(3):
            for k in range(3):
                mer_datalog[image_name]["patch-population"][i, j, k] = np.sum(object[j*increment:(j+1)*increment, k*increment:(k+1)*increment])
with open("mer_datalog.pickle", "wb") as f:
    pickle.dump(mer_datalog, f, protocol = pickle.HIGHEST_PROTOCOL)

# --- MSL Directory --- 
print("MSL Directory")
image_dir = ROOT / "msl" / "images" / "edr"
label_dir = ROOT / "msl" / "labels" / "train" # separate 'MSL' and 'MER' directories due to structure

for image_path in tqdm(list(label_dir.glob("*.png"))): # train directory
    msl_datalog[image_path.stem] = {}
    msl_datalog[image_path.stem]["patch-population"] = np.zeros((4, 3, 3))
    msl_datalog[image_path.stem]["object-centriod"] = np.zeros((4, 2))

    image_label = np.array(Image.open(image_path).convert("RGB"))[:, :, 0] # store color channel -- all are the same
    soil, bedrock, sand, bigrock = np.zeros(image_label.shape), np.zeros(image_label.shape), np.zeros(image_label.shape), np.zeros(image_label.shape)

    soil[np.where(image_label == 0)] = 1
    bedrock[np.where(image_label == 1)] = 1
    sand[np.where(image_label == 2)] = 1
    bigrock[np.where(image_label == 3)] = 1

    soil_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(soil == 1)).T) if np.concatenate(np.where(soil == 1)).size else -1
    bedrock_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(bedrock == 1)).T) if np.concatenate(np.where(bedrock == 1)).size else -1
    sand_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(sand == 1)).T) if np.concatenate(np.where(sand == 1)).size else -1
    bigrock_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(bigrock == 1)).T) if np.concatenate(np.where(bigrock == 1)).size else -1

    soil_center = np.flip(soil_mean.cluster_centers_[0], axis = -1) if soil_mean != -1 else np.array([-1, -1])
    bedrock_center = np.flip(bedrock_mean.cluster_centers_[0], axis = -1) if bedrock_mean != -1 else np.array([-1, -1])
    sand_center = np.flip(sand_mean.cluster_centers_[0], axis = -1) if sand_mean != -1 else np.array([-1, -1])
    bigrock_center = np.flip(bigrock_mean.cluster_centers_[0], axis = -1) if bigrock_mean != -1 else np.array([-1, -1])
    msl_datalog[image_path.stem]["object-centriod"] = np.vstack((soil_center, bedrock_center, sand_center, bigrock_center)) # row 0 -> soil, 1 -> bedrock ... 

    if visualize:
        fig, axes = plt.subplots(2, 3)
        image = np.array(Image.open(image_dir / (image_path.stem + ".JPG")).convert("RGB"))

        axes[0][0].imshow(image)
        axes[0][0].set_title("image")

        axes[0][1].imshow(soil)
        axes[0][1].set_title("soil")
        axes[0][1].scatter(soil_center[0], soil_center[1], marker = '*', c = 'k')

        axes[0][2].imshow(bedrock)
        axes[0][2].set_title("bedrock")
        axes[0][2].scatter(bedrock_center[0], bedrock_center[1], marker = '*', c = 'k')

        axes[1][0].imshow(sand)
        axes[1][0].set_title("sand")
        axes[1][0].scatter(sand_center[0], sand_center[1], marker = '*', c = 'k')

        axes[1][1].imshow(bigrock)
        axes[1][1].set_title("big rock")
        axes[1][1].scatter(bigrock_center[0], bigrock_center[1], marker = '*', c = 'k')
        plt.show()

    assert image_label.shape[0] == image_label.shape[1]
    increment = image_label.shape[0] // 3
    for i, object in enumerate([soil, bedrock, sand, bigrock]):
        for j in range(3):
            for k in range(3):
                msl_datalog[image_path.stem]["patch-population"][i, j, k] = np.sum(object[j*increment:(j+1)*increment, k*increment:(k+1)*increment])

with open("msl_datalog.pickle", "wb") as f:
    pickle.dump(msl_datalog, f, protocol = pickle.HIGHEST_PROTOCOL)