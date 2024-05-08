import torch
import numpy as np
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode

class GridImage:

    def __init__(self, image, rover_mask, range_mask, label, class_labels, gridsize=(5,5), scene_mask_threshold=.75):
        self.image = image
        self.rover_mask = rover_mask
        self.range_mask = range_mask
        self.label = label
        self.class_labels = class_labels
        self.gridsize = gridsize
        self.scene_mask_threshold = scene_mask_threshold

        self.grid = self._create_grid()
        self.obstacle_grid = self.grid == 0  

        self.upsample = Resize(self.image.shape[1:], interpolation=InterpolationMode.NEAREST_EXACT)

    def _create_grid(self):
        grid = []

        height_step = int(self.image.shape[1] / self.gridsize[0])
        width_step = int(self.image.shape[2] / self.gridsize[1])

        for i in range(self.gridsize[0]):
            for j in range(self.gridsize[1]):
                label_patch = self.label[:,i * height_step : (i+1) * height_step, j*width_step:(j+1)*width_step]
                rover_patch = self.rover_mask[i * height_step : (i+1) * height_step, j*width_step:(j+1)*width_step]
                range_patch = self.range_mask[i * height_step : (i+1) * height_step, j*width_step:(j+1)*width_step]

                label_counts = torch.sum(label_patch, (1,2))
                label = torch.argmax(label_counts)
                
                rover_count = torch.sum(rover_patch)
                range_count = torch.sum(range_patch)

                rover = rover_count / (height_step * width_step) >= self.scene_mask_threshold
                inrange = range_count / (height_step * width_step) >= self.scene_mask_threshold

                grid_center = (
                    int(i * height_step + height_step / 2),
                    int(j * width_step + width_step / 2)
                )
                grid.append({
                    "label": label,
                    "rover": rover,
                    "range": inrange,
                    "center": grid_center
                })  
        grid = np.array(grid).reshape(self.gridsize)
        return grid
    
    def _grid_mask(self, key):
        grid_label = np.array([
            grid_element[key] 
            for grid_element in self.grid.flatten()
        ]).reshape(self.grid.shape)
        return grid_label
    
    @property
    def grid_label(self):
        return self._grid_mask("label")
    
    @property
    def grid_rover(self):
        return self._grid_mask("rover")
    
    @property
    def grid_range(self):
        return self._grid_mask("range")
    
    def apply_mask(self, grid_mask):
        image_processed = self.image.squeeze(0).numpy().copy()
        if grid_mask is not None:
            grid_upsampled = self.upsample(torch.Tensor(grid_mask).unsqueeze(0)).squeeze(0).numpy()
            image_processed *= grid_upsampled
        image_plottable = image_processed * 255
        return image_plottable
    
    def get_class_mask(self, class_label):
        grid = self.grid_label == self.class_labels[class_label]
        return grid
    
    def get_path_pixel_coordinates(self, path):
        coordinates = [self.grid[point]["center"] for point in path]
        return np.array(coordinates)
    
    def get_obstacle_grid(self, obstacle_classes):
        free_space_grids = [
            self.get_class_mask(class_label) 
            for class_label in self.class_labels.keys() 
            if class_label not in obstacle_classes
        ]
        free_space_grid = sum(free_space_grids) 

        free_space_grid *= self.grid_rover == False
        free_space_grid *= self.grid_range == False
        return free_space_grid
        