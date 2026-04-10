#!/usr/bin/env python3
"""
MIA 2026 Project 1 — Evaluation Script
=======================================
Evaluates student submissions for the fundus image registration and vessel
segmentation project (EN 520.433/623).

Three tasks are evaluated:
  Task 1 (Grouping):       Accuracy of test-to-anchor assignment     -> ACC
  Task 2 (Registration):   MSE of warped coordinate predictions      -> MSE
  Task 3 (Segmentation):   Vessel segmentation on partial labels     -> TPR, TNR

Usage
---------------------------
  # Self-evaluation with the example test set (validation set):
  python evaluate.py --pred_dir <your_submission> --gt_dir validation_set/ground_truth

  # Final evaluation with the test set:
  python evaluate.py --pred_dir <your_submission> --gt_dir test_set/ground_truth

  # Show per-image / per-anchor details:
  python evaluate.py --pred_dir <your_submission> --gt_dir validation_set/ground_truth -v

Submission folder structure
---------------------------
Your submission folder should mirror the ground truth layout:

  <submission>/
  ├── correspondence/
  │   └── grouping.csv                          # Task 1
  ├── warped_coordinates/
  │   ├── test_01_warped_coordinates.csv        # Task 2
  │   ├── test_02_warped_coordinates.csv
  │   └── ...
  └── segmentation/
      ├── anchor_01.png                         # Task 3
      ├── anchor_01.npz
      ├── anchor_02.png
      ├── anchor_02.npz
      └── ...

File format details:

  grouping.csv
    Two columns with header row: test_id, anchor_id
    Example:
        test_id,anchor_id
        test_01,anchor_04
        test_02,anchor_02
        ...

  test_XX_warped_coordinates.csv
    Two columns with header row: x, y
    100 rows (one per sampled coordinate), values rounded to 2 decimal places.
    Example:
        x,y
        1626.77,1141.20
        1047.53,2490.50
        ...

  anchor_XX.png
    Single-channel binary vessel segmentation mask (for visual inspection).
    Pixel values: 0 = background, 255 = vessel.
    Image size must match the anchor image (e.g. 2912 x 2912).
    Note: .png files are NOT used for scoring; only .npz files are evaluated.

  anchor_XX.npz  (used for scoring)
    Compressed NumPy archive with key 'mask'.
    Contains a 2D binary array: 0 = background, 1 = vessel.
    Same spatial dimensions as the corresponding .png.

Ground truth folder structure (provided by us)
-----------------------------------------------
  <ground_truth>/
  ├── correspondence/
  │   └── grouping.csv
  ├── warped_coordinates/
  │   ├── test_01_warped_coordinates.csv
  │   └── ...
  └── mask/
      ├── vessel_mask_01.tiff                   # annotated vessel regions
      ├── bg_mask_01.tiff                       # annotated background regions
      └── ...

Evaluation metrics
------------------
  ACC = correctly assigned test images / total test images

  MSE = (1/MN) * sum_i sum_j [ (x_pred - x_gt)^2 + (y_pred - y_gt)^2 ]
        where M = number of test images, N = 100 sampled points

  TPR = predicted vessel pixels in annotated vessel-mask regions
        / total annotated vessel-mask pixels

  TNR = predicted background pixels in annotated background-mask regions
        / total annotated background-mask pixels

  Note: segmentation evaluation uses partial annotations. Only pixels inside
  the provided vessel_mask and bg_mask regions are evaluated; all other pixels
  are ignored.
"""

import argparse
import csv
import os
import sys
import numpy as np


def load_csv_coords(csv_path):
    """Load (x, y) coordinates from a CSV file with header 'x,y'."""
    coords = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append((float(row['x']), float(row['y'])))
    return np.array(coords, dtype=np.float64)


def load_grouping(csv_path):
    """Load grouping CSV into {test_id: anchor_id} dict."""
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['test_id'].strip()] = row['anchor_id'].strip()
    return mapping


def load_mask_npz(path):
    """Load a binary mask from .npz (returns bool array)."""
    data = np.load(path)
    for key in ['mask', 'vessel', 'arr_0', 'data', 'seg']:
        if key in data:
            return data[key] > 0
    keys = list(data.keys())
    if len(keys) == 1:
        return data[keys[0]] > 0
    raise ValueError(f"Cannot determine mask key in {path}. Available keys: {keys}. "
                     f"Please use 'mask' as the key name.")


def load_gt_mask(path):
    """Load a ground truth mask from .tiff (returns bool array)."""
    from PIL import Image
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img > 0


def evaluate_task1_grouping(pred_dir, gt_dir):
    """Evaluate anchor assignment accuracy."""
    gt_path = os.path.join(gt_dir, 'correspondence', 'grouping.csv')
    pred_path = os.path.join(pred_dir, 'correspondence', 'grouping.csv')

    if not os.path.isfile(gt_path):
        return None, "Ground truth grouping.csv not found"
    if not os.path.isfile(pred_path):
        return None, "Prediction grouping.csv not found"

    gt_map = load_grouping(gt_path)
    pred_map = load_grouping(pred_path)

    gt_tests = sorted(gt_map.keys())
    n_total = len(gt_tests)
    if n_total == 0:
        return None, "Ground truth grouping is empty"

    correct = 0
    missing = []
    details = []

    for test_id in gt_tests:
        if test_id not in pred_map:
            missing.append(test_id)
            details.append((test_id, gt_map[test_id], 'MISSING', False))
        else:
            is_correct = pred_map[test_id] == gt_map[test_id]
            if is_correct:
                correct += 1
            details.append((test_id, gt_map[test_id], pred_map[test_id], is_correct))

    acc = correct / n_total
    result = {
        'accuracy': acc,
        'correct': correct,
        'total': n_total,
        'missing': missing,
        'details': details,
    }
    return result, None


def evaluate_task2_registration(pred_dir, gt_dir):
    """Evaluate warped coordinate prediction MSE."""
    gt_warp_dir = os.path.join(gt_dir, 'warped_coordinates')
    pred_warp_dir = os.path.join(pred_dir, 'warped_coordinates')

    if not os.path.isdir(gt_warp_dir):
        return None, "Ground truth warped_coordinates/ directory not found"
    if not os.path.isdir(pred_warp_dir):
        return None, "Prediction warped_coordinates/ directory not found"

    gt_files = sorted([f for f in os.listdir(gt_warp_dir) if f.endswith('_warped_coordinates.csv')])
    if not gt_files:
        return None, "No ground truth warped coordinate files found"

    per_image_mse = []
    missing = []
    errors = []

    for gt_file in gt_files:
        test_id = gt_file.replace('_warped_coordinates.csv', '')
        pred_file = os.path.join(pred_warp_dir, gt_file)
        gt_file_path = os.path.join(gt_warp_dir, gt_file)

        if not os.path.isfile(pred_file):
            missing.append(test_id)
            continue

        try:
            gt_coords = load_csv_coords(gt_file_path)
            pred_coords = load_csv_coords(pred_file)

            if gt_coords.shape != pred_coords.shape:
                errors.append(
                    f"{test_id}: shape mismatch (pred={pred_coords.shape}, gt={gt_coords.shape})")
                continue

            diff = pred_coords - gt_coords
            mse_i = np.mean(diff[:, 0]**2 + diff[:, 1]**2)
            per_image_mse.append((test_id, mse_i))

        except Exception as e:
            errors.append(f"{test_id}: {e}")

    if not per_image_mse:
        return None, "No valid warped coordinate predictions to evaluate"

    M = len(per_image_mse)
    overall_mse = sum(mse for _, mse in per_image_mse) / M

    result = {
        'mse': overall_mse,
        'n_evaluated': M,
        'n_total': len(gt_files),
        'missing': missing,
        'errors': errors,
        'per_image': per_image_mse,
    }
    return result, None


def evaluate_task3_segmentation(pred_dir, gt_dir):
    """Evaluate vessel segmentation using partial annotations (TPR & TNR)."""
    gt_mask_dir = os.path.join(gt_dir, 'mask')

    if not os.path.isdir(gt_mask_dir):
        return None, "Ground truth mask/ directory not found (Task 3 skipped)"

    pred_seg_dir = os.path.join(pred_dir, 'segmentation')
    if not os.path.isdir(pred_seg_dir):
        return None, "Prediction segmentation/ directory not found"

    vessel_files = sorted([f for f in os.listdir(gt_mask_dir)
                           if f.startswith('vessel_mask_') and f.endswith('.tiff')])
    if not vessel_files:
        return None, "No vessel_mask_XX.tiff files found in ground truth"

    per_anchor = []
    missing = []
    errors = []

    total_tp = 0
    total_vessel_pixels = 0
    total_tn = 0
    total_bg_pixels = 0

    for vf in vessel_files:
        idx_str = vf.replace('vessel_mask_', '').replace('.tiff', '')
        anchor_id = f'anchor_{idx_str}'

        vessel_gt_path = os.path.join(gt_mask_dir, vf)
        bg_gt_path = os.path.join(gt_mask_dir, f'bg_mask_{idx_str}.tiff')

        pred_npz = os.path.join(pred_seg_dir, f'{anchor_id}.npz')

        if not os.path.isfile(pred_npz):
            missing.append(anchor_id)
            continue

        try:
            vessel_gt = load_gt_mask(vessel_gt_path)
            has_bg = os.path.isfile(bg_gt_path)
            bg_gt = load_gt_mask(bg_gt_path) if has_bg else None

            pred_mask = load_mask_npz(pred_npz)

            if pred_mask.shape != vessel_gt.shape:
                errors.append(
                    f"{anchor_id}: shape mismatch "
                    f"(pred={pred_mask.shape}, gt={vessel_gt.shape})")
                continue

            n_vessel = np.sum(vessel_gt)
            tp = np.sum(pred_mask[vessel_gt]) if n_vessel > 0 else 0
            tpr = tp / n_vessel if n_vessel > 0 else float('nan')

            total_tp += tp
            total_vessel_pixels += n_vessel

            n_bg = np.sum(bg_gt) if bg_gt is not None else 0
            tn = np.sum(~pred_mask[bg_gt]) if (bg_gt is not None and n_bg > 0) else 0
            tnr = tn / n_bg if n_bg > 0 else float('nan')

            total_tn += tn
            total_bg_pixels += n_bg

            per_anchor.append({
                'anchor_id': anchor_id,
                'tpr': tpr,
                'tnr': tnr,
                'tp': int(tp),
                'vessel_pixels': int(n_vessel),
                'tn': int(tn),
                'bg_pixels': int(n_bg),
            })

        except Exception as e:
            errors.append(f"{anchor_id}: {e}")

    if not per_anchor:
        return None, "No valid segmentation predictions to evaluate"

    overall_tpr = total_tp / total_vessel_pixels if total_vessel_pixels > 0 else float('nan')
    overall_tnr = total_tn / total_bg_pixels if total_bg_pixels > 0 else float('nan')

    result = {
        'tpr': overall_tpr,
        'tnr': overall_tnr,
        'total_tp': int(total_tp),
        'total_vessel_pixels': int(total_vessel_pixels),
        'total_tn': int(total_tn),
        'total_bg_pixels': int(total_bg_pixels),
        'n_evaluated': len(per_anchor),
        'n_total': len(vessel_files),
        'missing': missing,
        'errors': errors,
        'per_anchor': per_anchor,
    }
    return result, None


def print_separator(char='=', width=72):
    print(char * width)


def print_results(grouping_result, reg_result, seg_result, verbose=False):
    """Print evaluation results in a formatted table."""

    print_separator()
    print("  MIA 2026 Project 1 — Evaluation Results")
    print_separator()

    print("\n  Task 1: Anchor Assignment (Grouping)")
    print_separator('-')
    if grouping_result is None:
        print("  [SKIPPED]")
    else:
        r = grouping_result
        print(f"  Accuracy:  {r['accuracy']:.4f}  ({r['correct']}/{r['total']})")
        if r['missing']:
            print(f"  WARNING:   Missing {len(r['missing'])} test image(s): {', '.join(r['missing'])}")
        if verbose:
            wrong = [(t, gt, pred) for t, gt, pred, ok in r['details'] if not ok]
            if wrong:
                print(f"\n  Incorrect assignments ({len(wrong)}):")
                for test_id, gt_anchor, pred_anchor in wrong:
                    print(f"    {test_id}:  predicted={pred_anchor},  correct={gt_anchor}")

    print(f"\n  Task 2: Warped Coordinate Prediction (Registration)")
    print_separator('-')
    if reg_result is None:
        print("  [SKIPPED]")
    else:
        r = reg_result
        print(f"  MSE:       {r['mse']:.4f}")
        print(f"  Evaluated: {r['n_evaluated']}/{r['n_total']} test images")
        if r['missing']:
            print(f"  WARNING:   Missing predictions for: {', '.join(r['missing'])}")
        if r['errors']:
            for err in r['errors']:
                print(f"  ERROR:     {err}")
        if verbose:
            print(f"\n  Per-image MSE:")
            for test_id, mse in r['per_image']:
                print(f"    {test_id}:  MSE = {mse:.4f}")

    print(f"\n  Task 3: Vessel Segmentation")
    print_separator('-')
    if seg_result is None:
        print("  [SKIPPED]")
    else:
        r = seg_result
        print(f"  TPR:       {r['tpr']:.4f}  ({r['total_tp']}/{r['total_vessel_pixels']} vessel pixels)")
        print(f"  TNR:       {r['tnr']:.4f}  ({r['total_tn']}/{r['total_bg_pixels']} background pixels)")
        print(f"  Evaluated: {r['n_evaluated']}/{r['n_total']} anchor images")
        if r['missing']:
            print(f"  WARNING:   Missing predictions for: {', '.join(r['missing'])}")
        if r['errors']:
            for err in r['errors']:
                print(f"  ERROR:     {err}")
        if verbose:
            print(f"\n  Per-anchor results:")
            for a in r['per_anchor']:
                print(f"    {a['anchor_id']}:  TPR = {a['tpr']:.4f}  "
                      f"({a['tp']}/{a['vessel_pixels']}),  "
                      f"TNR = {a['tnr']:.4f}  ({a['tn']}/{a['bg_pixels']})")

    print()
    print_separator()
    print("  Summary")
    print_separator('-')

    acc_str = f"{grouping_result['accuracy']:.4f}" if grouping_result else "N/A"
    mse_str = f"{reg_result['mse']:.4f}" if reg_result else "N/A"
    tpr_str = f"{seg_result['tpr']:.4f}" if seg_result else "N/A"
    tnr_str = f"{seg_result['tnr']:.4f}" if seg_result else "N/A"

    print(f"  ACC  = {acc_str}")
    print(f"  MSE  = {mse_str}")
    print(f"  TPR  = {tpr_str}")
    print(f"  TNR  = {tnr_str}")
    print_separator()
    print()


def main():
    parser = argparse.ArgumentParser(
        description='MIA 2026 Project 1 — Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred_dir', required=True,
                        help='Path to the submission folder')
    parser.add_argument('--gt_dir', required=True,
                        help='Path to the ground truth folder')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show per-image and per-anchor details')
    parser.add_argument('--skip', nargs='+', choices=['grouping', 'registration', 'segmentation'],
                        default=[], help='Skip specific tasks')
    args = parser.parse_args()

    if not os.path.isdir(args.pred_dir):
        print(f"ERROR: Prediction directory not found: {args.pred_dir}")
        sys.exit(1)
    if not os.path.isdir(args.gt_dir):
        print(f"ERROR: Ground truth directory not found: {args.gt_dir}")
        sys.exit(1)

    grouping_result = None
    reg_result = None
    seg_result = None
    grouping_msg = None
    reg_msg = None
    seg_msg = None

    if 'grouping' not in args.skip:
        grouping_result, grouping_msg = evaluate_task1_grouping(args.pred_dir, args.gt_dir)
        if grouping_msg:
            print(f"  Task 1 note: {grouping_msg}")

    if 'registration' not in args.skip:
        reg_result, reg_msg = evaluate_task2_registration(args.pred_dir, args.gt_dir)
        if reg_msg:
            print(f"  Task 2 note: {reg_msg}")

    if 'segmentation' not in args.skip:
        seg_result, seg_msg = evaluate_task3_segmentation(args.pred_dir, args.gt_dir)
        if seg_msg:
            print(f"  Task 3 note: {seg_msg}")

    print_results(grouping_result, reg_result, seg_result, verbose=args.verbose)


if __name__ == '__main__':
    main()
