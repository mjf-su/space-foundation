
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import tqdm
import json

from msl_data import MSLData
from image_utils import *
from prompt_planner import *

plt.rc('font', size=20)
plt.rc('font', family='Times New Roman')
# plt.rc('fontname', "Times New Roman")
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'

DATA_DIR = pathlib.Path("data/ai4mars-dataset-merged-0.3/msl")
SAVE_DIR = pathlib.Path("data/planning")
dataset = MSLData(DATA_DIR)
N_IMAGES = 1000

gridsize = (5,5)
row_labels = ["top", "upper", "center", "lower", "bottom"]
column_labels = ["far left", "left", "center", "right", "far right"]
cardinal_directions = ["forward", "to the right", "backward", "to the left"]
class_aliasses = {
    "soil": "soil",
    "bedrock": "bedrock",
    "sand": "sand",
    "bigrock": "big rocks"
}

planner = Planner(
    gridsize, 
    row_labels, 
    column_labels, 
    cardinal_directions, 
    class_aliasses
)

total_samples = []

row_goal_points = [0,1,2,3]
col_goal_points = [0,1,2,3,4]

obstacle_classes_templates = [
    ["sand", "bigrock"],
    ["bigrock"],
    ["sand"]
]

def major_occlusions(mask):
    majority_rover_check = np.sum(mask) < .25 * np.prod(mask.shape)
    rover_in_top = np.any(np.sum(mask == 0, axis=1)[1:3])
    return majority_rover_check or rover_in_top

def is_valid_goal(goal, mask):
    return mask[goal[0], goal[1]] == 1

def edit_start_coord(start_coordinate, mask):
    start_coord = start_coordinate
    col_id =start_coord[1]
    return (max(np.where(mask[:,col_id] == 1)[0]), col_id)


for sample in tqdm.tqdm(range(N_IMAGES)):
    try:
        image, rover_mask, range_mask, label, raw_label, image_id = dataset[sample]
    except:
        print("image {} has no labels...".format(dataset.image_ids[sample]))
        continue

    grid_image = GridImage(
        image, 
        rover_mask, 
        range_mask, 
        label, 
        dataset.class_labels
    )
    navigable_area_mask = grid_image.get_obstacle_grid([])
    if not major_occlusions(navigable_area_mask):
        start_coord = edit_start_coord(planner.start_coordinate, navigable_area_mask)
        for row_goal in row_goal_points:
            for col_goal in col_goal_points:
                goal = (row_goal, col_goal)
                if is_valid_goal(goal, navigable_area_mask) and not start_coord == goal:
                    obstacle_classes = obstacle_classes_templates[np.random.randint(low=0, high=len(obstacle_classes_templates))]
                    samples = planner.generate_search_prompt(
                        grid_image, 
                        start_coord, 
                        goal, 
                        obstacle_classes,
                        source_dir=dataset.image_dir,
                        save_dir=SAVE_DIR,
                        image_id=image_id
                    )
                    total_samples += samples

with open(SAVE_DIR / "labels.json", 'w', encoding='utf-8') as f:
    json.dump(total_samples, f, ensure_ascii=False, indent=4)
