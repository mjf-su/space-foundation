import numpy as np
import pathlib
import pickle
from PIL import Image
from tqdm import tqdm

ROOT = pathlib.Path('TODO') # FILL IN WITH PATH TO DIRECTORY CONTAINING "ai4mars-dataset"
mer_image_dir = ROOT / "ai4mars-dataset" / "mer" / "images" / "eff"
mer_label_dir = ROOT / "ai4mars-dataset" / "mer" / "labels" / "train" / "merged-unmasked"

input_filename = "../preprocessed-data/mer_datalog_with_bbox.pickle"
output_filename = "../preprocessed-data/mer_datalog_with_vqa.pickle"

with open(input_filename, "rb") as f:
    mer_data = pickle.load(f)

for image_name in tqdm(mer_data):
    terrain_classes = ['soil', 'bedrock', 'sand', 'bigrock']
    vertical_region_str = ['top', 'middle', 'bottom']
    horizontal_region_str = ['left', 'middle', 'right']
    patch_coord_to_str = {} # assumes image split into 3x3 grid of patches
    for r, row_str in enumerate(vertical_region_str):
        for c, col_str in enumerate(horizontal_region_str):
            patch_coord_to_str[(r, c)] = f'{row_str}-{col_str}'

    with Image.open(next(mer_image_dir.glob(image_name + "*"))) as img:
        img_width, img_height = img.size # assumes single channel image
    preprocessed_img_width = 336 # based on CLIP visual encoder preprocessing
    preprocessed_img_height = 336 # based on CLIP visual encoder preprocessing

    patch_population = mer_data[image_name]['patch-population']
    object_centroids = mer_data[image_name]['object-centriods']
    object_centroids = np.round(object_centroids * np.array([(preprocessed_img_width - 1) / (img_width - 1), (preprocessed_img_height - 1) / (img_height - 1)]), 3) # assumes square image
    bounding_boxes = mer_data[image_name]['bounding-boxes']
    for i in range(len(bounding_boxes)):
        if len(bounding_boxes[i]) > 0:
            bounding_boxes[i] = np.round(bounding_boxes[i] * np.array([(preprocessed_img_height - 1) / (img_height - 1), (preprocessed_img_width - 1) / (img_width - 1), (preprocessed_img_height - 1) / (img_height - 1), (preprocessed_img_width - 1) / (img_width - 1)]), 3) # assumes square image

    vqa_data = {}

    # General (Terrain description)
    terrain_desription = {}
    classes_in_image = sorted([terrain_classes[i] for i in range(len(patch_population)) if np.sum(patch_population[i]) > 0])
    classes_not_in_image = sorted([terrain_classes[i] for i in range(len(patch_population)) if np.sum(patch_population[i]) == 0])
    terrain_desription['What terrain classes are present in this image? Provide your answer in alphabetical order.'] = f'The image contains the following terrain classes: {", ".join(classes_in_image)}.' if len(classes_in_image) > 0 else 'The image does not contain any terrain classes.'
    terrain_desription['Identify which of the following terrain classes are found in this image: bedrock, bigrock, sand, soil. Provide your answer in alphabetical order.'] = f'{", ".join(classes_in_image)} are found in the image.' if len(classes_in_image) > 1 else (f'{classes_in_image[0]} is found in the image.' if len(classes_in_image) > 0 else 'None of those terrain classes are found in the image.')
    terrain_desription['What terrain classes are not present in this image? Provide your answer in alphabetical order.'] = f'The image does not contain the following terrain classes: {", ".join(classes_not_in_image)}.' if len(classes_not_in_image) > 0 else 'The image is not missing any terrain classes.'
    terrain_desription['Identify which of the following terrain classes are not found in this image: bedrock, bigrock, sand, soil. Provide your answer in alphabetical order.'] = f'{", ".join(classes_not_in_image)} are not found in the image.' if len(classes_not_in_image) > 1 else (f'{classes_not_in_image[0]} is not found in the image.' if len(classes_not_in_image) > 0 else 'All of those terrain classes are found in the image.')
    terrain_desription['What is the most prevalent terrain class in this image?'] = f'The most prevalent terrain class in the image is {terrain_classes[np.argmax(np.sum(patch_population, axis=(1,2)))]}.' if len(classes_in_image) > 0 else 'The image does not contain any terrain classes.'
    for i, terrain_class in enumerate(terrain_classes):
        terrain_desription[f'What percentage of this image is occupied by {terrain_class}?'] = f'{np.round(100 * np.sum(patch_population[i]) / (img_height * img_width), 3)} percent of the image is occupied by {terrain_class}.'
    vqa_data['terrain-description'] = terrain_desription

    # Object Localization (Single object localization)
    single_object_localization = {}
    for i, terrain_class in enumerate(terrain_classes):
        single_object_localization[f'Identify an instance of {terrain_class} in the image.'] = f'An example of {terrain_class} can be found at {object_centroids[i]} in the image.' if np.sum(patch_population[i]) > 0 else f'The image does not contain any {terrain_class}.'
        single_object_localization[f'Identify the point around which most of the {terrain_class} is centered in this image.'] = f'Most of the {terrain_class} in the image is centered around {object_centroids[i]}.' if np.sum(patch_population[i]) > 0 else f'The image does not contain any {terrain_class}.'
        single_object_localization[f'What coordinate best represents the distribution of {terrain_class} in this image?'] = f'The distribution of {terrain_class} in the image is best represented by its centroid {object_centroids[i]}.' if np.sum(patch_population[i]) > 0 else f'The image does not contain any {terrain_class}.'
    vqa_data['single-object-localization'] = single_object_localization

    # Object Localization (Multi-instance localization)
    multi_instance_localization = {}
    for i, terrain_class in enumerate(terrain_classes):
        regions_with_class = sorted([patch_coord_to_str[tuple(patch_coord)] for patch_coord in np.transpose(np.nonzero(patch_population[i] > 0))])
        regions_without_class = sorted([patch_coord_to_str[tuple(patch_coord)] for patch_coord in np.transpose(np.nonzero(patch_population[i] == 0))])
        multi_instance_localization[f'What regions of this image contain {terrain_class}? Provide your answer in alphabetical order.'] = f'The following regions of the image contain some presence of {terrain_class}: {", ".join(regions_with_class)}' if len(regions_with_class) > 0 else f'There are no regions of the image that contain any {terrain_class}.'
        multi_instance_localization[f'What regions of this image do not contain {terrain_class}? Provide your answer in alphabetical order.'] = f'The following regions of the image do not contain any presence of {terrain_class}: {", ".join(regions_without_class)}' if len(regions_without_class) > 0 else f'All the regions in the image contain some presence of {terrain_class}.'
        multi_instance_localization[f'How many significant clusters of {terrain_class} are in this image?'] = f'There are {len(bounding_boxes[i])} significant clusters of {terrain_class} in the image.' if len(bounding_boxes[i]) != 1 else f'There is 1 significant cluster of {terrain_class} in the image.'
    vqa_data['multi-instance-localization'] = multi_instance_localization

    # Object Localization (Relative object localization)
    relative_object_localization = {}
    for i, terrain_class_1 in enumerate(terrain_classes):
        for j, terrain_class_2 in enumerate(terrain_classes):
            if terrain_class_2 == terrain_class_1:
                continue
            if np.sum(patch_population[i]) == 0:
                relative_object_localization[f'Where is the {terrain_class_2} located relative to the {terrain_class_1} in this image?'] = f'There is no {terrain_class_1} in the image.'
                relative_object_localization[f'Is the {terrain_class_2} located to the left or right of the {terrain_class_1} in this image?'] = f'There is no {terrain_class_1} in the image.'
                relative_object_localization[f'Is the {terrain_class_2} located above or below the {terrain_class_1} in this image?'] = f'There is no {terrain_class_1} in the image.'
                relative_object_localization[f'From the image-taker\'s perspective, is the {terrain_class_2} in front of or behind the {terrain_class_1}?'] = f'There is no {terrain_class_1} in the image.'
                continue
            if np.sum(patch_population[j]) == 0:
                relative_object_localization[f'Where is the {terrain_class_2} located relative to the {terrain_class_1} in this image?'] = f'There is no {terrain_class_2} in the image.'
                relative_object_localization[f'Is the {terrain_class_2} located to the left or right of the {terrain_class_1} in this image?'] = f'There is no {terrain_class_2} in the image.'
                relative_object_localization[f'Is the {terrain_class_2} located above or below the {terrain_class_1} in this image?'] = f'There is no {terrain_class_2} in the image.'
                relative_object_localization[f'From the image-taker\'s perspective, is the {terrain_class_2} in front of or behind the {terrain_class_1}?'] = f'There is no {terrain_class_2} in the image.'
                continue
            above_or_below = 'same'
            front_or_behind = 'same'
            if object_centroids[j][1] < object_centroids[i][1]:
                above_or_below = 'above'
                front_or_behind = 'behind'
            elif object_centroids[j][1] > object_centroids[i][1]:
                above_or_below = 'below'
                front_or_behind = 'in front of'
            left_or_right = 'same'
            if object_centroids[j][0] < object_centroids[i][0]:
                left_or_right = 'to the left of'
            elif object_centroids[j][0] > object_centroids[i][0]:
                left_or_right = 'to the right of'
            relative_location = [relation for relation in [above_or_below, left_or_right] if relation != 'same']
            relative_object_localization[f'Where is {terrain_class_2} located relative to {terrain_class_1} in this image?'] = f'The {terrain_class_2} is primarily located {" and ".join(relative_location)} the {terrain_class_1} in the image.' if len(relative_location) > 0 else f'The {terrain_class_2} is primarily located in the same location as the {terrain_class_1} in the image.'
            relative_object_localization[f'Is the {terrain_class_2} located to the left or right of the {terrain_class_1} in this image?'] = f'The {terrain_class_2} is mostly {left_or_right} the {terrain_class_1} in the image.' if left_or_right != 'same' else f'The {terrain_class_2} mostly has the same horizontal position as the {terrain_class_1} in the image.'
            relative_object_localization[f'Is the {terrain_class_2} located above or below the {terrain_class_1} in this image?'] = f'The {terrain_class_2} is mostly {above_or_below} the {terrain_class_1} in the image.' if above_or_below != 'same' else f'The {terrain_class_2} mostly has the same vertical position as the {terrain_class_1} in the image.'
            relative_object_localization[f'From the image-taker\'s perspective, is the {terrain_class_2} in front of or behind the {terrain_class_1}?'] = f'The {terrain_class_2} is mostly {front_or_behind} the {terrain_class_1} from the perspective of the image-taker.' if above_or_below != 'same' else f'The {terrain_class_2} mostly has the same depth position as the {terrain_class_1} from the perspective of the image-taker.'
    closest_class = None
    for i, bbox in enumerate(bounding_boxes):
        if len(bbox) == 0:
            continue
        if closest_class is None or np.max(bbox[:, 2]) > closest_class[1]:
            closest_class = (terrain_classes[i], np.max(bbox[:, 2]))
    relative_object_localization[f'If you were to move forward in the direction that this image was taken, what terrain class would you cross first?'] = f'By moving forward in the direction that the image was taken, we would first cross {closest_class[0]}.' if closest_class is not None else 'There is not enough information to determine what terrain class we would cross first by moving forward in the direction that the image was taken.'
    vqa_data['relative-object-localization'] = relative_object_localization

    # Object Description (Object coverage)
    object_coverage = {}
    for i, terrain_class in enumerate(terrain_classes):
        centroid_vertical_loc = 'top' if object_centroids[i][1] < img_height / 3 else ('bottom' if object_centroids[i][1] > 2 * img_height / 3 else 'middle')
        centroid_horizontal_loc = 'left' if object_centroids[i][0] < img_width / 3 else ('right' if object_centroids[i][0] > 2 * img_width / 3 else 'middle')
        regions_with_class = sorted([patch_coord_to_str[tuple(patch_coord)] for patch_coord in np.transpose(np.nonzero(patch_population[i] > 0))])
        object_coverage[f'How is {terrain_class} spread out throughout this image? Provide any lists in alphabetical order.'] = f'The {terrain_class} is centered in the {centroid_vertical_loc}-{centroid_horizontal_loc} of the image, and has a presence in the {", ".join(regions_with_class)} of the image.' if np.sum(patch_population[i]) > 0 else f'There is no {terrain_class} in the image.'
        patch_percentages = np.round(100 * patch_population[i] / (np.sum(patch_population[i])), 3) if np.sum(patch_population[i]) > 0 else np.zeros_like(patch_population[i])
        distribution = [f'the {patch_coord_to_str[(r,c)]} contains {patch_percentages[r, c]} percent of the {terrain_class}' for r in range(len(patch_population[i])) for c in range(len(patch_population[i][r]))]
        object_coverage[f'What is the distribution of {terrain_class} in this image?'] = f'In the image, the {terrain_class} is distributed as follows: {", ".join(distribution)}.' if np.sum(patch_population[i]) > 0 else f'There is no {terrain_class} in the image.'
        object_coverage[f'What region of this image contains the most {terrain_class}?'] = f'The {patch_coord_to_str[np.unravel_index(np.argmax(patch_population[i]), patch_population[i].shape)]} of the image contains the most {terrain_class}.' if np.sum(patch_population[i]) > 0 else f'There is no {terrain_class} in the image.'
        object_coverage[f'Is the majority of the {terrain_class} found in the top, middle, or bottom of this image?'] = f'The majority of the {terrain_class} is located in the {vertical_region_str[np.argmax(np.sum(patch_population[i], axis=1))]} of the image' if np.sum(patch_population[i]) > 0 else f'There is no {terrain_class} in the image.'
        object_coverage[f'Is the majority of the {terrain_class} found in the left, middle, or right of this image?'] = f'The majority of the {terrain_class} is located in the {horizontal_region_str[np.argmax(np.sum(patch_population[i], axis=0))]} of the image' if np.sum(patch_population[i]) > 0 else f'There is no {terrain_class} in the image.'
    vqa_data['object-coverage'] = object_coverage

    # Object Description (Object size)
    object_size = {}
    for i, terrain_class in enumerate(terrain_classes):
        bbox_areas = np.round([(maxr - minr) * (maxc - minc) for minr, minc, maxr, maxc in bounding_boxes[i]], 3)
        object_size[f'What is the area of the largest cluster of {terrain_class} in this image?'] = f'The largest cluster of {terrain_class} in the image takes up an area of {np.max(bbox_areas)} square pixels.' if len(bounding_boxes[i]) > 0 else f'There are no significant clusters of {terrain_class} in the image.'
        object_size[f'What is the area of the smallest cluster of {terrain_class} in this image?'] = f'The smallest cluster of {terrain_class} in the image takes up an area of {np.min(bbox_areas)} square pixels.' if len(bounding_boxes[i]) > 0 else f'There are no significant clusters of {terrain_class} in the image.'
        object_size[f'What is the average area of a cluster of {terrain_class} in this image?'] = f'A cluster of {terrain_class} in the image takes up an average area of {np.round(np.mean(bbox_areas), 3)} square pixels.' if len(bounding_boxes[i]) > 0 else f'There are no significant clusters of {terrain_class} in the image.'
    vqa_data['object-size'] = object_size

    # Object Description (Object traversability)
    object_traversability = {}
    total_population = np.sum(patch_population, axis=0)
    total_population[total_population == 0] = 1
    safe_regions = sorted([patch_coord_to_str[tuple(patch_coord)] for patch_coord in np.transpose(np.nonzero((patch_population[0] + patch_population[1]) / total_population > 0.5))])
    object_traversability['What regions of this image can a vehicle safely drive into? Provide your answer in alphabetical order.'] = f'A vehicle can safely drive into the following regions of the image since they contain mostly bedrock and/or soil: {", ".join(safe_regions)}' if len(safe_regions) > 0 else 'Based on this image, there are no regions containing mostly bedrock and/or soil that a vehicle can safely drive into.'
    for i, bbox_list in enumerate(bounding_boxes):
        for bbox in bbox_list:
            object_traversability[f'Is the terrain bounded by [min_row, min_column, max_row, max_column] = {list(bbox)} in this image generally safe for a vehicle to drive over?'] = f'The terrain bounded by {list(bbox)} in the image {"is" if (terrain_classes[i] == "bedrock" or terrain_classes[i] == "soil") else "is not"} generally safe for a vehicle to drive over since it contains mostly {terrain_classes[i]}.'
            object_traversability[f'Would a vehicle need to avoid driving over the terrain contained in [min_row, min_column, max_row, max_column] = {list(bbox)} in this image?'] = f'A vehicle {"would" if (terrain_classes[i] != "bedrock" and terrain_classes[i] != "soil") else "would not"} need to avoid driving over the terrain within {list(bbox)} in the image since it contains mostly {terrain_classes[i]}.'
    vqa_data['object-traversability'] = object_traversability

    mer_data[image_name]['vqa-data'] = vqa_data

with open(output_filename, "wb") as f:
    pickle.dump(mer_data, f)