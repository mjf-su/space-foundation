import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.dijkstra import DijkstraFinder

plt.rc('font', size=20)
plt.rc('font', family='Times New Roman')
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'

def stringify_label(start, goal, obstacle_classes):
    start_str = str(start[0]) + str(start[1])
    goal_str = str(goal[0]) + str(goal[1])
    obstacle_string = "_".join(obstacle_classes)
    return start_str + "_to_" + goal_str + "_avoiding_" + obstacle_string

def format_template(template, key, value):
    return template.replace(key, value)


def sample_generation(*args):
    generation = " ".join([
        np.random.choice(arg) for arg in args
    ])
    return generation

def format_sample(id, image_path, input, response):
    prompt = {
        "id": id,
        "image": image_path,
        "conversations" : [
            {
                "from": "human",
                "value": input
            },
            {
                "from": "gpt",
                "value": response
            }
        ]
    }
    return prompt


class Planner:

    def __init__(self, gridsize, row_labels, column_labels, cardinal_directions, class_aliasses):
        self.gridsize = gridsize
        self.row_labels = row_labels
        self.column_labels = column_labels
        self.cardinal_directions = cardinal_directions
        self.finder = DijkstraFinder(diagonal_movement=DiagonalMovement.never)
        self.start_coordinate = [gridsize[0] - 1, int(gridsize[1] / 2)]
        self.class_aliasses = class_aliasses

        self.keys = {
            "goal": "{Goal}",
            "obstacles": "{Obstacles}",
            "plan" : "{Plan}"
        }
        self.prompt_template_texts = {
            "navigation_primer": [
                "This is an onboard camera image <image> from a Mars rover.",
                "See the attached rover dashcam image <image>.",
                "Based on the provided picture <image> onboard the robot:"
            ],
            "navigation_overlay_primer": [
                "This is an onboard camera image <image> from a Mars rover. On the image, a proposed motion plan for the robot has been overlayed."
            ],
            "feasibility": [
                "Is there a feasible path for the rover to drive towards the {Goal} region in the image while avoiding any {Obstacles}?",
                "Is it possible for the rover to reach the {Goal} region in the image without passing over any {Obstacles}?"
            ],
            "planning": [
                "Describe step-by-step how the rover should navigate to reach the {Goal} region in the image without driving over {Obstacles}.",
                "Provide a high-level stepwise plan for the rover to navigate to the {Goal} region in the image while avoiding any {Obstacles}.",
                "Identify a sequence of motion commands for the rover in order to travel to the {Goal} region in the image. The rover musn't pass over {Obstacles}."
            ],
            "planning_no_obstacles": [
                "Describe how the rover should navigate to reach the {Goal} region in the image.",
                "Provide a high-level plan for the rover to navigate to the {Goal} region in the image.",
                "Identify a sequence of motion commands for the rover in order to travel to the {Goal} region in the image."
            ],
            "overlay_feasibility" : [
                "Does the provided path reach the {Goal} region in the image while avoiding any {Obstacles}?",
                "Is this a feasible plan to reach the {Goal} region in the image without passing over any {Obstacles}?",
                "Does the given plan avoid driving over any {Obstacles}?"
            ]
        }
        self.response_template_texts = {
            "feasibility": {
                "true": ["Yes, it is possible for the rover to drive to the {Goal} region while avoiding any {Obstacles}."],
                "false": ["No, it is not possible for the rover to drive to the {Goal} region while avoiding any {Obstacles}."]
            },
            "overlay_feasibility": {
                "true": ["Yes, the provided path reaches the {Goal} region while avoiding any {Obstacles}."],
                "false": ["No, the given plan does not lead to to the {Goal} region while avoiding {Obstacles}."]
            },
            "planning" : [
                "To reach the {Goal} region in the image and avoid {Obstacles}, the rover should drive {Plan}.",
                "The rover should drive {Plan}. This will reach the {Goal} region without driving over {Obstacles}."
            ],
            "planning_no_obstacles" : [
                "To reach the {Goal} region in the image, the rover should drive {Plan}. This will avoid {Obstacles}, which pose hazards to the rover.",
                "The rover should drive {Plan}. This will reach the {Goal} region without driving over {Obstacles}, which are unsafe for the rover."
            ]
        }

    def _plan_coordinate_path(self, start, end, mask):
        grid = Grid(matrix=mask)
        start = grid.node(start[1], start[0])
        end = grid.node(end[1], end[0])
        path, _ = self.finder.find_path(start, end, grid)
        path = [(point.y, point.x) for point in path]
        return path

    def annotate_plan_on_image(self, grid_image, plan, file=None, mask=None):
        coords = grid_image.get_path_pixel_coordinates(plan)
        image = grid_image.apply_mask(mask)
        
        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.plot(coords[:,1], coords[:,0], "white", marker="o", lw=4)
        txt = plt.text(coords[0,1], coords[0,0], "start", color="tab:blue", fontweight="bold", horizontalalignment="center", verticalalignment="top")
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        txt = plt.text(coords[-1,1], coords[-1,0], "goal", color="tab:red", fontweight="bold", horizontalalignment="center", verticalalignment="center")
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        plt.axis("off")
        plt.tight_layout()
        if file:
            plt.savefig(file, bbox_inches="tight", pad_inches=0)
        plt.close()


    def get_plan(self, grid_image, start, end, obstacle_classes, save_dir=None, image_id=None, image_label_suffix=None):
        mask = grid_image.get_obstacle_grid(obstacle_classes)
        plan = self._plan_coordinate_path(start, end, mask)

        if len(plan) != 0:
            annotation_plan = plan
        else: 
            annotation_plan = self._plan_coordinate_path(start, end, np.ones(mask.shape))

        if save_dir is not None and image_id is not None and image_label_suffix is not None:
            self.annotate_plan_on_image(
                grid_image, 
                annotation_plan, 
                file=str(save_dir / "unmasked" / (image_id + "_" + image_label_suffix + ".png"))
            )
            
            self.annotate_plan_on_image(
                grid_image, 
                annotation_plan, 
                file=str(save_dir / "masked" / (image_id + "_" + image_label_suffix + ".png")),
                mask=mask
            )
        else: 
            self.annotate_plan_on_image(grid_image, annotation_plan)
            self.annotate_plan_on_image(grid_image, annotation_plan, mask=mask)

        return plan
    
    def _plan_to_text(self, plan):
        plan = np.array(plan)
        movements = np.diff(plan, axis=0)
        textplan = []
        for movement in movements:
            if np.all(movement == np.array([-1,0])): #North East South West
                textplan.append(self.cardinal_directions[0])
            if np.all(movement == np.array([0,1])):
                textplan.append(self.cardinal_directions[1])
            elif np.all(movement == np.array([1,0])):
                textplan.append(self.cardinal_directions[2])
            elif np.all(movement == np.array([0,-1])):
                textplan.append(self.cardinal_directions[3])

        if len(textplan) == 1:
            return textplan[0]
        elif len(textplan) == 2:
            return textplan[0] + " and then " + textplan[1]
        else:
            text = ", then ".join(textplan[:-1])
            text = text + ", and finally " + textplan[-1]
            return text
    
    def _goal_to_text(self, goal):
        row_str = self.row_labels[goal[0]]
        col_str = self.column_labels[goal[1]]

        if row_str == "center" and col_str == "center":
            return row_str
        else:
            return col_str + ", " + row_str
        
    def _obstacle_to_text(self, obstacles):
        obstacles = [self.class_aliasses[obstacle] for obstacle in obstacles]
        if len(obstacles) > 1:
            obstacles[-1] = "and " + obstacles[-1]
            if len(obstacles) > 2:
                return ", ".join(obstacles)
            else:
                return " ".join(obstacles)
        else:
            return obstacles[0]
        
    def _compile_template(self, template, goal=None, obstacle_classes=None, plan=None):
        template = format_template(template, self.keys["goal"], goal)
        template = format_template(template, self.keys["obstacles"], obstacle_classes)
        if plan is not None:
            template = format_template(template, self.keys["plan"], plan)
        return template
    
    def generate_search_prompt(
            self, 
            grid_image, 
            start, 
            end, 
            obstacle_classes, 
            source_dir=None, 
            save_dir=None, 
            image_id=None
        ):

        image_label_suffix = stringify_label(start, end, obstacle_classes)

        plan = self.get_plan(
            grid_image, 
            start, 
            end, 
            obstacle_classes, 
            save_dir=save_dir,
            image_id=image_id, 
            image_label_suffix=image_label_suffix
        )

        # If there is a feasible path: generate 3 prompts
        #   Base feasibility question
        #   Granular motion plan
        #   Scoring an annotated image
        # if there is no feasible path: generate 2 prompts
        #   Base feasibility question
        #   Scoring an annotated image
        
        goal_text = self._goal_to_text(end)
        obstacle_text = self._obstacle_to_text(obstacle_classes)

        feasibility_prompt = self._compile_template(
            sample_generation(
                self.prompt_template_texts["navigation_primer"],
                self.prompt_template_texts["feasibility"]
            ),
            goal=goal_text,
            obstacle_classes=obstacle_text
        )
        feasibility_overlay_prompt = self._compile_template(
            sample_generation(
                self.prompt_template_texts["navigation_overlay_primer"],
                self.prompt_template_texts["overlay_feasibility"]
            ),
            goal=goal_text,
            obstacle_classes=obstacle_text
        )

        prompts = []
    
        if len(plan) > 0:
            plan_text = self._plan_to_text(plan)

            planning_prompt = self._compile_template(
                sample_generation(
                    self.prompt_template_texts["navigation_primer"],
                    self.prompt_template_texts["planning"]
                ),
                goal=goal_text,
                obstacle_classes=obstacle_text,
                plan=plan_text
            )
            planning_response = self._compile_template(
                sample_generation(
                    self.response_template_texts["planning"]
                ),
                goal=goal_text,
                obstacle_classes=obstacle_text,
                plan=plan_text
            )

            planning_sample = format_sample(
                "_".join((image_id, image_label_suffix, "planning")),
                str(source_dir / (image_id + ".JPG")),
                planning_prompt,
                planning_response
            )
            prompts.append(planning_sample)

            feasibility_response = self._compile_template(
                sample_generation(
                    self.response_template_texts["feasibility"]["true"]
                ),
                goal=goal_text,
                obstacle_classes=obstacle_text,
                plan=plan_text
            )

            feasibility_overlay_response = self._compile_template(
                sample_generation(
                    self.response_template_texts["overlay_feasibility"]["true"]
                ),
                goal=goal_text,
                obstacle_classes=obstacle_text,
                plan=plan_text
            )

        else:
            feasibility_response = self._compile_template(
                sample_generation(
                    self.response_template_texts["feasibility"]["false"]
                ),
                goal=goal_text,
                obstacle_classes=obstacle_text
            )

            feasibility_overlay_response = self._compile_template(
                sample_generation(
                    self.response_template_texts["overlay_feasibility"]["false"]
                ),
                goal=goal_text,
                obstacle_classes=obstacle_text
            )

        feasibility_sample = format_sample(
                "_".join((image_id, image_label_suffix, "feasibility")),
                str(source_dir / (image_id + ".JPG")),
                feasibility_prompt,
                feasibility_response
        )
        prompts.append(feasibility_sample)
        feasibility_overlay_sample = format_sample(
                "_".join((image_id, image_label_suffix, "feasibility_overlay")),
                str(save_dir / "unmasked" / (image_id + "_" + image_label_suffix + ".png")),
                feasibility_overlay_prompt,
                feasibility_overlay_response
        )
        prompts.append(feasibility_overlay_sample)
        return prompts
        
    #todo
    #implement a function that takes two grid coordinates, the obstacle types, and plans the path.
    # returns the path in text
    # also takes the path, and creates an annotated image with the path overlayed
    # returns the annotated image even if there is no valid path

    #todo
    #implement a function that takes two grid coordinates, obstacle types, and returns a series of prompts, and annotated images
      
