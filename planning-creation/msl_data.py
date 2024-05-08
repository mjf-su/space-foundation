from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
from torch.utils.data import Dataset
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms import ToTensor, ToPILImage
plt.rc('font', size=20)
plt.rc('font', family='Times New Roman')
# plt.rc('fontname', "Times New Roman")
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'

class MSLData(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.class_labels = {
            "soil":0,
            "bedrock":1,
            "sand":2,
            "bigrock":3,
            "null":255
        }
        self.annotation_colors = {
            "soil": "green",
            "bedrock": "cyan",
            "sand": "orange",
            "bigrock": "magenta",
            "rover": "red",
            "range limit": "blue" 
        }

        self.image_dir = self.data_dir / "images" / "edr"
        self.rover_mask_dir = self.data_dir / "images" / "mxy"
        self.range_mask_dir = self.data_dir / "images" / "rng-30m"
        self.label_dir = self.data_dir / "labels" / "train"

        self.image_ids = [img.stem for img in self.image_dir.glob("*.JPG")]

        self.tf = ToTensor()
        self.invtf = ToPILImage()

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(self.image_dir / (image_id + ".JPG"))
        rover_mask = Image.open(self.rover_mask_dir / (image_id.replace("EDR", "MXY") + ".png"))
        range_mask = Image.open(self.range_mask_dir / (image_id.replace("EDR", "RNG") + ".png"))
        label = Image.open(self.label_dir / (image_id + ".png"))

        image = self.tf(image)
        rover_mask = torch.Tensor(np.array(rover_mask) == 1)
        range_mask = torch.Tensor(np.array(range_mask) == 1)
        raw_label = torch.Tensor(np.array(label))
        label = torch.stack(
            (
                raw_label == self.class_labels["soil"],
                raw_label == self.class_labels["bedrock"],
                raw_label == self.class_labels["sand"],
                raw_label == self.class_labels["bigrock"]
            )
        )
        return image, rover_mask, range_mask, label, raw_label, image_id
    
    def plot_sample(self, idx, figsize=(9,7)):
        image, rover_mask, range_mask, label, raw_label, image_id = self.__getitem__(idx)
        segment_image = draw_segmentation_masks(
            (image * 255).expand(3, -1, -1).to(torch.uint8),
            torch.cat((label, rover_mask.unsqueeze(0), range_mask.unsqueeze(0)), dim=0).to(torch.bool), 
            alpha=.3, 
            colors=[
                self.annotation_colors["soil"],
                self.annotation_colors["bedrock"],
                self.annotation_colors["sand"],
                self.annotation_colors["bigrock"],
                self.annotation_colors["rover"],
                self.annotation_colors["range limit"]
            ]
        )
        plt.figure(figsize=figsize)
        proxy_patches = [Patch(color=color, label=colorlabel) for colorlabel, color in self.annotation_colors.items()]
        plt.imshow(self.invtf(segment_image))
        plt.axis("off")
        plt.legend(handles=[*proxy_patches], loc="center right", bbox_to_anchor=(1.6,.5))
        return plt.gca()
    

        






