from PIL import Image
import pathlib
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
image1 = "1n179678560eff6000p1920l0m1" #OG
image2 = "1n207287541eff74v1p1825l0m1" #non-OG
ROOT = pathlib.Path().cwd()
test_label_path = pathlib.Path(ROOT / "space-foundation" / "planning-creation" / "data" / "ai4mars-dataset-merged-0.3" / "mer" / "labels" / "test" / "raw" / "1n207287541eff74v1p1825l0m1_16165_T0_SOL891_2145455.png")
test_image_path = pathlib.Path(ROOT / "space-foundation" / "planning-creation" / "data" / "ai4mars-dataset-merged-0.3/msl/images/edr/NLB_531373164EDR_F0590936NCAM07753M1.JPG")

print("test image exists: ", test_image_path.exists())

test_label = np.array(Image.open(test_label_path))
test_image = np.array(Image.open(test_image_path).convert('RGB'))

soil, bedrock, sand, bigrock = np.zeros(test_label.shape), np.zeros(test_label.shape), np.zeros(test_label.shape), np.zeros(test_label.shape)

soil[np.where(test_label == 0)] = 1
bedrock[np.where(test_label == 1)] = 1
sand[np.where(test_label == 2)] = 1
bigrock[np.where(test_label == 3)] = 1

soil_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(soil == 1)).T) if np.concatenate(np.where(soil == 1)).size else -1
bedrock_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(bedrock == 1)).T) if np.concatenate(np.where(bedrock == 1)).size else -1
sand_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(sand == 1)).T) if np.concatenate(np.where(sand == 1)).size else -1
bigrock_mean = KMeans(n_clusters=1, init = "random", n_init = 100).fit(np.vstack(np.where(bigrock == 1)).T) if np.concatenate(np.where(bigrock == 1)).size else -1

soil_center = np.flip(soil_mean.cluster_centers_[0], axis = -1) if soil_mean != -1 else np.array([-1, -1])
bedrock_center = np.flip(bedrock_mean.cluster_centers_[0], axis = -1) if bedrock_mean != -1 else np.array([-1, -1])
sand_center = np.flip(sand_mean.cluster_centers_[0], axis = -1) if sand_mean != -1 else np.array([-1, -1])
bigrock_center = np.flip(bigrock_mean.cluster_centers_[0], axis = -1) if bigrock_mean != -1 else np.array([-1, -1])

edge = 1024 // 5

fig, axes = plt.subplots(2, 3)
axes[0][0].imshow(test_image)
axes[0][0].axhline(edge)
axes[0][0].axhline(2*edge)
axes[0][0].axhline(3*edge)
axes[0][0].axhline(4*edge)
axes[0][0].axvline(edge)
axes[0][0].axvline(2*edge)
axes[0][0].axvline(3*edge)
axes[0][0].axvline(4*edge)

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

plt.imshow(test_image)
plt.axhline(edge)
plt.axhline(2*edge)
plt.axhline(3*edge)
plt.axhline(4*edge)
plt.axvline(edge)
plt.axvline(2*edge)
plt.axvline(3*edge)
plt.axvline(4*edge)
plt.show()