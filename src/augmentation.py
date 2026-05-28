"""
augmentation.py
---------------
Core image augmentation utilities for the garment hole detection pipeline.
Handles hue shifting, rotation with bounding box transformation, and
COCO-format dataset augmentation to address class imbalance.
"""

import cv2
import numpy as np
import json
import os
import random
from pathlib import Path
from copy import deepcopy
from collections import Counter


def adjust_hue(image: np.ndarray, hue_shift: int) -> np.ndarray:
    """
    Shift the hue channel of a BGR image.

    Args:
        image: BGR image as numpy array
        hue_shift: Integer shift value (-179 to 179)

    Returns:
        Hue-shifted BGR image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.int16) + hue_shift
    h = np.clip(h, 0, 179).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def rotate_image_and_bbox(
    image: np.ndarray,
    bboxes: list,
    angle: float,
    img_w: int,
    img_h: int
) -> tuple:
    """
    Rotate image and transform COCO bounding boxes accordingly.

    Args:
        image: BGR image
        bboxes: List of [x, y, w, h] COCO-format bounding boxes
        angle: Rotation angle in degrees
        img_w: Image width
        img_h: Image height

    Returns:
        Tuple of (rotated_image, transformed_bboxes)
    """
    center = (img_w / 2, img_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (img_w, img_h))

    new_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        points = np.hstack([points, np.ones((4, 1))])
        transformed = (M @ points.T).T
        x_min = transformed[:, 0].min()
        y_min = transformed[:, 1].min()
        x_max = transformed[:, 0].max()
        y_max = transformed[:, 1].max()
        new_bboxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

    return rotated, new_bboxes


def load_coco_annotations(json_path: str) -> dict:
    """Load a COCO-format annotation JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def augment_dataset(
    input_image_dir: str,
    input_json_path: str,
    output_image_dir: str,
    output_json_path: str,
    target_with_holes: int = 750,
    target_without_holes: int = 750,
    hue_shifts: list = None,
    angles: list = None,
    seed: int = 42
) -> dict:
    """
    Augment a COCO-format dataset to a balanced target size using
    hue shifts and rotations.

    Args:
        input_image_dir: Directory containing original images
        input_json_path: Path to COCO annotations JSON
        output_image_dir: Directory to save augmented images
        output_json_path: Path to save augmented COCO JSON
        target_with_holes: Target number of images with hole annotations
        target_without_holes: Target number of images without annotations
        hue_shifts: List of hue shift values (default: ±10, ±20, ±30)
        angles: List of rotation angles (default: ±20, ±30)
        seed: Random seed for reproducibility

    Returns:
        New COCO data dict with augmented images and annotations
    """
    random.seed(seed)
    np.random.seed(seed)

    if hue_shifts is None:
        hue_shifts = [-30, -20, -10, 10, 20, 30]
    if angles is None:
        angles = [-30, -20, 20, 30]

    Path(output_image_dir).mkdir(parents=True, exist_ok=True)

    coco_data = load_coco_annotations(input_json_path)
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Identify hole category (most common)
    category_counts = Counter(ann['category_id'] for ann in annotations)
    hole_cat_id = category_counts.most_common(1)[0][0]

    image_ids_with_holes = set(ann['image_id'] for ann in annotations)
    images_with_holes = [img for img in images if img['id'] in image_ids_with_holes]
    images_without_holes = [img for img in images if img['id'] not in image_ids_with_holes]

    ann_per_image = Counter(ann['image_id'] for ann in annotations)

    new_images = []
    new_annotations = []
    new_image_id = max(img['id'] for img in images) + 1
    new_ann_id = max(ann['id'] for ann in annotations) + 1

    orig_with = len(images_with_holes)
    orig_without = len(images_without_holes)
    base_aug_with = (target_with_holes - orig_with) / orig_with
    base_aug_without = (target_without_holes - orig_without) / orig_without

    def _augment_group(image_list, target, has_holes, base_aug):
        nonlocal new_image_id, new_ann_id
        count = len(image_list)

        for img in image_list:
            img_path = os.path.join(input_image_dir, img['file_name'])
            image = cv2.imread(img_path)
            if image is None:
                continue

            bboxes = []
            img_anns = []
            if has_holes:
                img_anns = [a for a in annotations if a['image_id'] == img['id']]
                bboxes = [[a['bbox'][0], a['bbox'][1], a['bbox'][2], a['bbox'][3]] for a in img_anns]

            num_holes = ann_per_image.get(img['id'], 0)
            if has_holes and num_holes > 3:
                num_aug = max(20, int(base_aug * 4))
            elif has_holes:
                num_aug = max(10, int(base_aug * 2))
            else:
                num_aug = int(base_aug) + (1 if random.random() < (base_aug % 1) else 0)

            for _ in range(num_aug):
                if count >= target:
                    break

                hue = random.choice(hue_shifts)
                angle = random.choice(angles)
                aug_img = adjust_hue(image, hue)
                aug_img, new_bboxes = rotate_image_and_bbox(aug_img, bboxes, angle, img['width'], img['height'])

                new_filename = f"aug_{new_image_id}_{img['file_name']}"
                save_path = os.path.join(output_image_dir, new_filename)
                if not cv2.imwrite(save_path, aug_img):
                    continue

                new_img_entry = deepcopy(img)
                new_img_entry['id'] = new_image_id
                new_img_entry['file_name'] = new_filename
                new_images.append(new_img_entry)

                for bbox, orig_ann in zip(new_bboxes, img_anns):
                    new_ann = deepcopy(orig_ann)
                    new_ann['id'] = new_ann_id
                    new_ann['image_id'] = new_image_id
                    new_ann['bbox'] = [float(v) for v in bbox]
                    new_annotations.append(new_ann)
                    new_ann_id += 1

                new_image_id += 1
                count += 1

    _augment_group(images_with_holes, target_with_holes, True, base_aug_with)
    _augment_group(images_without_holes, target_without_holes, False, base_aug_without)

    new_coco_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories
    }

    with open(output_json_path, 'w') as f:
        json.dump(new_coco_data, f, indent=2)

    print(f"Augmented dataset: {len(new_images)} images, {len(new_annotations)} annotations")
    return new_coco_data
