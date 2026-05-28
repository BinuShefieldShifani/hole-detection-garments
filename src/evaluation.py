"""
evaluation.py
-------------
Production evaluation utilities for the garment hole detection pipeline.
Supports both COCO JSON and YOLO TXT annotation formats.
"""

import json
import cv2
import numpy as np
from pathlib import Path


class ProductionEvaluator:
    """
    Full production evaluator supporting YOLO-format annotations.
    Computes Precision, Recall, F1, Detection Rate, and Empty Precision.
    """

    def __init__(self, iou_threshold: float = 0.5, conf_threshold: float = 0.10):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def load_annotations(self, test_dir: str) -> tuple:
        """Load YOLO-format .txt annotation files from a directory."""
        test_path = Path(test_dir)
        annotations = {}
        image_files = []

        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(test_path.glob(ext))

        total_holes = 0
        images_with_holes = 0

        for img_path in image_files:
            txt_path = test_path / f"{img_path.stem}.txt"
            gt_boxes = []

            if txt_path.exists():
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            xc, yc, w, h = map(float, parts[1:5])
                            if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1:
                                gt_boxes.append([xc, yc, w, h])

            annotations[img_path.stem] = gt_boxes
            if gt_boxes:
                images_with_holes += 1
                total_holes += len(gt_boxes)

        print(f"Loaded {len(image_files)} images | "
              f"{images_with_holes} with holes | "
              f"{total_holes} total GT holes")

        return annotations, image_files

    def calculate_iou(self, box1: list, box2: list) -> float:
        """Calculate IoU between two [x1, y1, x2, y2] absolute boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def evaluate(self, agent, test_dir: str, agent_name: str = "Agent") -> dict:
        """
        Run full evaluation of an agent on a test directory.

        Args:
            agent: Object with a .predict(image_path) method returning
                   {'boxes': np.ndarray, 'scores': np.ndarray, 'labels': np.ndarray}
            test_dir: Path to directory with images and .txt label files
            agent_name: Display name for logging

        Returns:
            Dictionary of evaluation metrics
        """
        annotations, image_files = self.load_annotations(test_dir)
        tp = fp = fn = 0
        images_with_detections = 0

        for img_path in image_files:
            pred = agent.predict(str(img_path))
            pred_boxes = pred.get('boxes', np.array([]))

            if len(pred_boxes) > 0:
                images_with_detections += 1

            gt_boxes_norm = annotations.get(img_path.stem, [])
            if len(gt_boxes_norm) == 0 and len(pred_boxes) == 0:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Convert normalised GT to absolute
            gt_abs = []
            for gt in gt_boxes_norm:
                xc, yc, bw, bh = gt
                x1 = max(0, (xc - bw / 2) * w)
                y1 = max(0, (yc - bh / 2) * h)
                x2 = min(w, (xc + bw / 2) * w)
                y2 = min(h, (yc + bh / 2) * h)
                if x2 > x1 and y2 > y1:
                    gt_abs.append([x1, y1, x2, y2])

            gt_abs = np.array(gt_abs)
            matched_gt = set()

            for pred_box in pred_boxes:
                best_iou, best_idx = 0, -1
                for j, gt_box in enumerate(gt_abs):
                    if j in matched_gt:
                        continue
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_idx = iou, j

                if best_iou > self.iou_threshold:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1

            fn += len(gt_abs) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        detection_rate = images_with_detections / len(image_files)
        empty_precision = 1.0 if fp == 0 else 1.0 - fp / len(image_files)

        results = {
            'agent_name': agent_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp, 'fp': fp, 'fn': fn,
            'total_gt': tp + fn,
            'images_evaluated': len(image_files),
            'images_with_detections': images_with_detections,
            'detection_rate': detection_rate,
            'empty_precision': empty_precision,
            'iou_threshold': self.iou_threshold,
            'conf_threshold': self.conf_threshold
        }

        self._print_results(results)
        return results

    def _print_results(self, r: dict):
        print(f"\n{'='*55}")
        print(f"  {r['agent_name']} — Evaluation Results")
        print(f"{'='*55}")
        print(f"  Precision      : {r['precision']:.3f}  ({r['precision']:.1%})")
        print(f"  Recall         : {r['recall']:.3f}  ({r['recall']:.1%})")
        print(f"  F1 Score       : {r['f1_score']:.3f}")
        print(f"  Detection Rate : {r['detection_rate']:.1%}")
        print(f"  Empty Precision: {r['empty_precision']:.3f}")
        print(f"  TP / FP / FN   : {r['tp']} / {r['fp']} / {r['fn']}")
        print(f"  Total GT Holes : {r['total_gt']}")
        print(f"{'='*55}\n")


class DebugEvaluator:
    """Lightweight evaluator for quick sanity checks on small samples."""

    def quick_test(self, agent, test_dir: str, sample_size: int = 10) -> dict:
        test_path = Path(test_dir)
        image_files = list(test_path.glob("*.png"))[:sample_size]
        tp = fp = fn = 0

        for img_path in image_files:
            txt_path = test_path / f"{img_path.stem}.txt"
            gt_boxes = []
            if txt_path.exists():
                with open(txt_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            gt_boxes.append(list(map(float, parts[1:5])))

            pred = agent.predict(str(img_path))
            pred_boxes = pred.get('boxes', [])
            print(f"{img_path.name}: GT={len(gt_boxes)}, Pred={len(pred_boxes)}")

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            gt_abs = []
            for gt in gt_boxes:
                x1 = max(0, (gt[0] - gt[2] / 2) * w)
                y1 = max(0, (gt[1] - gt[3] / 2) * h)
                x2 = min(w, (gt[0] + gt[2] / 2) * w)
                y2 = min(h, (gt[1] + gt[3] / 2) * h)
                gt_abs.append([x1, y1, x2, y2])

            matched = set()
            for pb in pred_boxes:
                best_iou, best_j = 0, -1
                for j, gb in enumerate(gt_abs):
                    if j in matched:
                        continue
                    x1 = max(pb[0], gb[0]); y1 = max(pb[1], gb[1])
                    x2 = min(pb[2], gb[2]); y2 = min(pb[3], gb[3])
                    if x2 > x1 and y2 > y1:
                        inter = (x2 - x1) * (y2 - y1)
                        union = ((pb[2]-pb[0])*(pb[3]-pb[1]) + (gb[2]-gb[0])*(gb[3]-gb[1]) - inter)
                        iou = inter / union if union > 0 else 0
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                if best_iou > 0.5:
                    tp += 1
                    matched.add(best_j)
                else:
                    fp += 1
            fn += len(gt_abs) - len(matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Quick Test: P={precision:.3f}, R={recall:.3f} | TP={tp}, FP={fp}, FN={fn}")
        return {'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn}
