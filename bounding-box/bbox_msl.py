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
msl_image_dir = ROOT / "ai4mars-dataset" / "msl" / "images" / "edr"
msl_label_dir = ROOT / "ai4mars-dataset" / "msl" / "labels" / "train"

input_filename = "../preprocessed-data/msl_datalog.pickle"
output_filename = "../preprocessed-data/msl_datalog_with_bbox.pickle"

visualize = '--visualize' in sys.argv

with open(input_filename, "rb") as file_id:
    msl_data = pickle.load(file_id)

for image_name in tqdm(list(msl_data.keys())):
    image_label = np.array(Image.open(msl_label_dir / (image_name + ".png")))
    soil, bedrock, sand, bigrock = np.zeros(image_label.shape), np.zeros(image_label.shape), np.zeros(image_label.shape), np.zeros(image_label.shape)

    soil[np.where(image_label == 0)] = 1
    bedrock[np.where(image_label == 1)] = 1
    sand[np.where(image_label == 2)] = 1
    bigrock[np.where(image_label == 3)] = 1

    image = np.array(Image.open(msl_image_dir / (image_name + ".JPG")))
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
    msl_data[image_name]['bounding-boxes'] = bounding_boxes

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
    pickle.dump(msl_data, file_id)