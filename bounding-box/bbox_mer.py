import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
import sys
import torch
from PIL import Image
from skimage import measure
from torchvision.ops import nms
from tqdm import tqdm

ROOT = pathlib.Path('TODO') # FILL IN WITH PATH TO DIRECTORY CONTAINING "ai4mars-dataset"
mer_image_dir = ROOT / "ai4mars-dataset" / "mer" / "images" / "eff"
mer_label_dir = ROOT / "ai4mars-dataset" / "mer" / "labels" / "train" / "merged-unmasked"

input_filename = "../preprocessed-data/mer_datalog.pickle"
output_filename = "../preprocessed-data/mer_datalog_with_bbox.pickle"

visualize = '--visualize' in sys.argv

with open(input_filename, "rb") as file_id:
    mer_data = pickle.load(file_id)

for image_name in tqdm(list(mer_data.keys())):
    reviews = []
    for labeled_image_path in mer_label_dir.glob(image_name + "*"):
        labeled_image = np.array(Image.open(labeled_image_path))
        soil, bedrock, sand, bigrock = np.zeros(labeled_image.shape), np.zeros(labeled_image.shape), np.zeros(labeled_image.shape), np.zeros(labeled_image.shape)

        soil[np.where(labeled_image == 0)] = 1
        bedrock[np.where(labeled_image == 1)] = 1
        sand[np.where(labeled_image == 2)] = 1
        bigrock[np.where(labeled_image == 3)] = 1

        reviews.append(np.stack((soil, bedrock, sand, bigrock)))

    meta_reviews = np.sum(np.stack(reviews), axis = 0)
    semantic_segmentation = np.argmax(meta_reviews, axis = 0)
    soil, bedrock, sand, bigrock = np.split(meta_reviews, 4)
    soil, bedrock, sand, bigrock = soil.squeeze(0), bedrock.squeeze(0), sand.squeeze(0), bigrock.squeeze(0)

    soil[np.where((semantic_segmentation != 0) + (soil < 3))] = 0 # zero non-dominant mask or mask without at least three reviewers in agreement
    bedrock[np.where((semantic_segmentation != 1) + (bedrock < 3))] = 0
    sand[np.where((semantic_segmentation != 2) + (sand < 3))] = 0
    bigrock[np.where((semantic_segmentation != 3) + (bigrock < 3))] = 0

    soil[np.where(soil > 0)] = 1
    bedrock[np.where(bedrock > 0)] = 1
    sand[np.where(sand > 0)] = 1
    bigrock[np.where(bigrock > 0)] = 1

    image = np.array(Image.open(next(mer_image_dir.glob(image_name + "*"))))
    class_masks = [soil, bedrock, sand, bigrock]
    bounding_boxes = []
    for i in range(len(class_masks)):
        connected_regions = measure.label(class_masks[i])
        region_props = measure.regionprops(connected_regions)
        region_props = [region for region in region_props if region.area_bbox >= 0.01 * np.prod(image.shape)]
        if len(region_props) == 0:
            bounding_boxes.append(np.array([]))
            continue
        boxes = np.array([np.array(region.bbox) for region in region_props])
        indices = nms(
            torch.stack([torch.Tensor(region.bbox)[[1, 0, 3, 2]] for region in region_props], dim=0),
            torch.Tensor([region.area_bbox for region in region_props]),
            0.01
        ).numpy()
        bounding_boxes.append(boxes[indices])
    mer_data[image_name]['bounding-boxes'] = bounding_boxes

    if visualize:
        classes = ['soil', 'bedrock', 'sand', 'bigrock']
        colors = ['red', 'yellow', 'green', 'blue']
        plt.figure()
        plt.imshow(image, cmap='grey')
        ax = plt.gca()
        for i in range(len(classes)):
            for j, bbox in enumerate(bounding_boxes[i]):
                minr, minc, maxr, maxc = bbox
                rect = patches.Rectangle(
                    (minc, minr),
                    maxc - minc,
                    maxr - minr,
                    fill=False,
                    edgecolor=colors[i],
                    linewidth=2,
                    label=classes[i] if j == 0 else None # prevent duplicate labels
                )
                ax.add_patch(rect)
        plt.legend()
        plt.show()

with open(output_filename, "wb") as file_id:
    pickle.dump(mer_data, file_id)