import json
import numpy as np
import pathlib
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = pathlib.Path('TODO') # FILL IN WITH PATH TO "ai4mars-dataset"

def create_train_val_json(datalog_filename, image_dir, mission_name):
    with open(datalog_filename, "rb") as f:
        data = pickle.load(f)

    full_image_dir = ROOT / image_dir
    train_images, val_images = train_test_split(list(data.keys()), train_size=0.8)
    train_samples = []
    val_samples = []
    for images, sample_list in [(train_images, train_samples), (val_images, val_samples)]:
        for image_name in tqdm(images):
            image_file = pathlib.Path(next(full_image_dir.glob(image_name + "*"))).name
            for question_category in data[image_name]['vqa-data']:
                question = np.random.choice(list(data[image_name]['vqa-data'][question_category].keys()))
                answer = data[image_name]['vqa-data'][question_category][question]
                sample_data = {
                    'id': f'{mission_name}-{image_name}-{question_category}',
                    'image': str(image_dir / image_file),
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<image>\n{question}'
                        },
                        {
                            'from': 'gpt',
                            'value': answer
                        },
                    ]
                }
                sample_list.append(sample_data)
    
    return train_samples, val_samples



train_samples = []
val_samples = []

mer_image_dir = pathlib.Path('mer/images/eff')
mer_filename = 'preprocessed-data/mer_datalog_with_vqa.pickle'
mer_train_samples, mer_val_samples = create_train_val_json(mer_filename, mer_image_dir, 'mer')
train_samples.extend(mer_train_samples)
val_samples.extend(mer_val_samples)

msl_image_dir = pathlib.Path('msl/images/edr')
msl_filename = 'preprocessed-data/msl_datalog_with_vqa.pickle'
msl_train_samples, msl_val_samples = create_train_val_json(msl_filename, msl_image_dir, 'msl')
train_samples.extend(msl_train_samples)
val_samples.extend(msl_val_samples)

with open('llava_ft_train.json', 'w') as f:
    json.dump(train_samples, f, indent=4)
with open('llava_ft_val.json', 'w') as f:
    json.dump(val_samples, f, indent=4)