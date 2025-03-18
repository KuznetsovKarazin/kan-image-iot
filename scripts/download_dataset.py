"""
Script for downloading and preparing Visual Wake Words dataset.
This script automates the process of downloading MSCOCO dataset
and creating Visual Wake Words annotations.

Author: Oleksandr Kuznetsov
Date: March 2025
"""

import os
import argparse
import subprocess
import sys
import requests
from tqdm import tqdm
from pathlib import Path

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def run_command(command):
    """Run shell command and return status"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Download and prepare Visual Wake Words dataset')
    parser.add_argument('--output_dir', type=str, default='data', 
                        help='Directory to store dataset')
    parser.add_argument('--year', type=str, default='2014',
                       help='MSCOCO dataset year (2014 or 2017)')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'processed'), exist_ok=True)
    
    coco_dir = os.path.join(args.output_dir, 'raw', 'coco')
    vww_dir = os.path.join(args.output_dir, 'processed', 'vww')
    
    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(vww_dir, exist_ok=True)
    os.makedirs(os.path.join(vww_dir, 'annotations'), exist_ok=True)
    
    # Step 1: Download MSCOCO dataset
    print(f"Step 1: Downloading MSCOCO {args.year} dataset to {coco_dir}")
    
    # Download annotations
    annotations_url = f"http://images.cocodataset.org/annotations/annotations_train{args.year}.zip"
    annotations_file = os.path.join(coco_dir, f"annotations_{args.year}.zip")
    
    if not os.path.exists(annotations_file):
        print(f"Downloading annotations from {annotations_url}")
        download_file(annotations_url, annotations_file)
    else:
        print(f"Annotations file already exists: {annotations_file}")
    
    # Download train images
    train_url = f"http://images.cocodataset.org/zips/train{args.year}.zip"
    train_file = os.path.join(coco_dir, f"train{args.year}.zip")
    
    if not os.path.exists(train_file):
        print(f"Downloading training images from {train_url}")
        download_file(train_url, train_file)
    else:
        print(f"Training images file already exists: {train_file}")
    
    # Download val images
    val_url = f"http://images.cocodataset.org/zips/val{args.year}.zip"
    val_file = os.path.join(coco_dir, f"val{args.year}.zip")
    
    if not os.path.exists(val_file):
        print(f"Downloading validation images from {val_url}")
        download_file(val_url, val_file)
    else:
        print(f"Validation images file already exists: {val_file}")
    
    # Step 2: Unzip files
    print("\nStep 2: Extracting files")
    
    # Extract annotations
    if not os.path.exists(os.path.join(coco_dir, 'annotations')):
        print(f"Extracting {annotations_file}")
        import zipfile
        with zipfile.ZipFile(annotations_file, 'r') as zip_ref:
            zip_ref.extractall(coco_dir)
    else:
        print("Annotations already extracted")
    
    # Extract train images
    if not os.path.exists(os.path.join(coco_dir, f'train{args.year}')):
        print(f"Extracting {train_file}")
        import zipfile
        with zipfile.ZipFile(train_file, 'r') as zip_ref:
            zip_ref.extractall(coco_dir)
    else:
        print("Training images already extracted")
    
    # Extract val images
    if not os.path.exists(os.path.join(coco_dir, f'val{args.year}')):
        print(f"Extracting {val_file}")
        import zipfile
        with zipfile.ZipFile(val_file, 'r') as zip_ref:
            zip_ref.extractall(coco_dir)
    else:
        print("Validation images already extracted")
    
    # Step 3: Create train-minival split
    print("\nStep 3: Creating COCO train-minival split")
    train_ann = os.path.join(coco_dir, f'annotations/instances_train{args.year}.json')
    val_ann = os.path.join(coco_dir, f'annotations/instances_val{args.year}.json')
    out_dir = os.path.join(coco_dir, 'annotations')
    
    cmd = f"python -m pyvww.utils.create_coco_train_minival_split --train_annotations_file=\"{train_ann}\" --val_annotations_file=\"{val_ann}\" --output_dir=\"{out_dir}\""
    run_command(cmd)
    
    # Step 4: Create Visual Wake Words annotations
    print("\nStep 4: Creating Visual Wake Words annotations")
    maxitrain_ann = os.path.join(coco_dir, 'annotations/instances_maxitrain.json')
    minival_ann = os.path.join(coco_dir, 'annotations/instances_minival.json')
    vww_ann_dir = os.path.join(vww_dir, 'annotations')
    
    cmd = f"python -m pyvww.utils.create_visualwakewords_annotations --train_annotations_file=\"{maxitrain_ann}\" --val_annotations_file=\"{minival_ann}\" --output_dir=\"{vww_ann_dir}\" --threshold=0.005 --foreground_class=\"person\""
    run_command(cmd)
    
    # Step 5: Create subset for easier experimentation
    print("\nStep 5: Creating subset for experimentation")
    subset_dir = os.path.join(args.output_dir, 'processed', 'vww_subset_large')
    subset_train = os.path.join(subset_dir, 'train')
    subset_val = os.path.join(subset_dir, 'val')
    
    os.makedirs(subset_train, exist_ok=True)
    os.makedirs(os.path.join(subset_train, 'person'), exist_ok=True)
    os.makedirs(os.path.join(subset_train, 'no_person'), exist_ok=True)
    
    os.makedirs(subset_val, exist_ok=True)
    os.makedirs(os.path.join(subset_val, 'person'), exist_ok=True)
    os.makedirs(os.path.join(subset_val, 'no_person'), exist_ok=True)
    
    print("Dataset preparation complete!")
    print(f"MSCOCO dataset: {coco_dir}")
    print(f"Visual Wake Words annotations: {vww_ann_dir}")
    print(f"Subset for experimentation: {subset_dir}")
    print("\nNext steps: Run the script to create the balanced subset for your experiments")

if __name__ == "__main__":
    main()