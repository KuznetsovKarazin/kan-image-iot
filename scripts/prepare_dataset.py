"""
Script for preparing Visual Wake Words dataset.
This script, given the vww annotations and the COCO dataset, creates a balanced subset
and creating Visual Wake Words annotations.
It copies the images in labeled folders and resizes them to a specified size.

The sets are:
- Training set: 50,000 (default) images with person + 50,000 images without person from maxitrain
- Validation set: 3,000 (default) images with person + 3,000 images without person from minival
- Test set: 3,000 (default) images with person + 3,000 images without person from maxitrain (excluding training images)

Before running this script:
ensure you have the COCO dataset downloaded and organized as follows:
data/raw/coco/
    ├── train2017/
    └── val2017/
    └── annotations/
Ensure that the annotations of vww are generated using the create_coco_train_minival.py script and
that you are also running the create_visual_wake_words_annotations.py script.

Author: Daniele Faggi
Date: September 2025
"""

import json
import os
from argparse import ArgumentParser
import random
from PIL import Image

def load_coco(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def get_coco_ids(coco_data):
    return set([img['id'] for img in coco_data['images']])

def get_vww_ids(coco_data):
    return set([ann['image_id'] for ann in coco_data['annotations'] if ann['category_id'] == 1])

def copy_resize_images(coco_data, selected_ids, input_dir, output_dir, size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    for img_id in selected_ids:
        filename = id_to_filename[img_id]
        parts = filename.split("_")
        folder = parts[1]
        src_path = os.path.join(input_dir, folder, filename)
        dst_path = os.path.join(output_dir, filename)
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                try:
                    img = Image.open(src_path).convert('RGB')
                    # Resize image according to resize type used in MobileNetV2 training
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    img.save(dst_path)
                except Exception as e:
                    print(f"Errore con {src_path}: {e}")        
        else:
            print(f"Warning: Source image does not exist: {src_path}")

def main(args):

    random.seed(args.random_seed)

    image_size = (int(args.image_size), int(args.image_size))

    # Where are the working directories?
    output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    input_dir = os.path.realpath(os.path.expanduser(args.input_dir))

    # Input vww annotations
    instances_maxitrain = os.path.join(input_dir, 'annotations/instances_maxitrain.json')
    instances_minival = os.path.join(input_dir, 'annotations/instances_minival.json')

    # Input person labeled vww images
    instances_maxitrain_person = os.path.join(output_dir, 'vww/annotations/instances_train.json')
    instances_minival_person = os.path.join(output_dir, 'vww/annotations/instances_val.json')

    # Output directories
    output_train = os.path.join(output_dir, 'vww_subset/train')
    output_val = os.path.join(output_dir, 'vww_subset/val')
    output_test = os.path.join(output_dir, 'vww_subset/test')

    # Start processing
    print("Preparing Visual Wake Words subset dataset...")

    # Load vww ids
    maxitrain_data = load_coco(instances_maxitrain)
    minival_data = load_coco(instances_minival)

    maxitrain_ids = get_coco_ids(maxitrain_data)
    minival_ids = get_coco_ids(minival_data)

    # Load person ids (from output dir)
    maxitrain_ids_person = get_vww_ids(load_coco(instances_maxitrain_person))
    minival_ids_person = get_vww_ids(load_coco(instances_minival_person))

    # Get no-person ids
    maxitrain_ids_no_person = maxitrain_ids.difference(maxitrain_ids_person)
    minival_ids_no_person = minival_ids.difference(minival_ids_person)

    print(f"Maxitrain: {len(maxitrain_ids)} total, {len(maxitrain_ids_person)} with person, {len(maxitrain_ids_no_person)} without person")
    print(f"Minival: {len(minival_ids)} total, {len(minival_ids_person)} with person, {len(minival_ids_no_person)} without person")

    # Sample images
    print(f"Sampling {args.train_samples} images per class for training")
    selected_train_person_ids = random.sample(list(maxitrain_ids_person), k=args.train_samples)
    selected_train_no_person_ids = random.sample(list(maxitrain_ids_no_person), k=args.train_samples)

    print(f"Sampling {args.validation_samples} images per class for validation")        
    selected_val_person_ids = random.sample(list(minival_ids_person), k=args.validation_samples)
    selected_val_no_person_ids = random.sample(list(minival_ids_no_person), k=args.validation_samples)

    # Extract test samples from maxitrain set (not in training set)
    print(f"Sampling {args.test_samples} images per class for final evaluation test set")
    test_ids_person = maxitrain_ids_person.difference(selected_train_person_ids)
    test_ids_no_person = maxitrain_ids_no_person.difference(selected_train_no_person_ids)
    selected_test_person_ids = random.sample(list(set(test_ids_person)), k=args.test_samples)
    selected_test_no_person_ids = random.sample(list(set(test_ids_no_person)), k=args.test_samples)

    # Copy and resize images
    print("Copying and resizing train images...")
    copy_resize_images(maxitrain_data, selected_train_person_ids, input_dir, os.path.join(output_train, 'person'), size=image_size)
    copy_resize_images(maxitrain_data, selected_train_no_person_ids, input_dir, os.path.join(output_train, 'no_person'), size=image_size)
    print("Copying and resizing validation images...")
    copy_resize_images(minival_data, selected_val_person_ids, input_dir, os.path.join(output_val, 'person'), size=image_size)
    copy_resize_images(minival_data, selected_val_no_person_ids, input_dir, os.path.join(output_val, 'no_person'), size=image_size)
    print("Copying and resizing test images...")
    copy_resize_images(maxitrain_data, selected_test_person_ids, input_dir, os.path.join(output_test, 'person'), size=image_size)
    copy_resize_images(maxitrain_data, selected_test_no_person_ids, input_dir, os.path.join(output_test, 'no_person'), size=image_size)

    print("Done.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=False,
                        default='data/processed',
                        help='Directory where the prepared dataset will be stored')
    parser.add_argument('--input_dir', type=str, required=False,
                        default='data/raw/coco',
                        help='COCO raw dataset directory containing maxitrain and minival annotations')
    parser.add_argument('--image_size', type=str, required=False,
                        default='224',
                        help='Target image size (e.g., 224)')
    parser.add_argument('--random_seed', type=int, required=False,
                        default=42,
                        help='Random seed for reproducibility')   
    parser.add_argument('--train_samples', type=int, required=False,
                        default=50000,
                        help='Training samples per class') 
    parser.add_argument('--validation_samples', type=int, required=False,
                        default=3000,
                        help='Validation samples per class')      
    parser.add_argument('--test_samples', type=int, required=False,
                        default=3000,
                        help='Final evaluation test samples per class from training dataset')
    args = parser.parse_args()
    main(args)

