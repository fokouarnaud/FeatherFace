#!/usr/bin/env python
"""
Debug script to understand the prediction dictionary structure
"""

import os
import numpy as np
import scipy.io as sio

def read_pred_file(filepath):
    """Read prediction file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    if lines:
        boxes = np.array(list(map(lambda x: [a for a in x.strip().split()], lines))).astype('float')
    else:
        boxes = np.array([])
    return img_file.split('/')[-1], boxes

def get_preds(pred_dir):
    """Load predictions"""
    events = os.listdir(pred_dir)
    boxes = dict()

    for event in events:
        event_dir = os.path.join(pred_dir, event)
        if not os.path.isdir(event_dir):
            continue

        event_images = os.listdir(event_dir)
        current_event = dict()

        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            key = imgname.rstrip('.jpg')
            current_event[key] = _boxes

        boxes[event] = current_event

    return boxes

def main():
    # Load predictions
    pred_dir = './widerface_evaluate/widerface_txt/'
    print(f"Loading predictions from: {pred_dir}")
    pred = get_preds(pred_dir)

    # Load ground truth
    gt_path = 'widerface_evaluate/eval_tools/ground_truth/'
    print(f"Loading ground truth from: {gt_path}")
    gt_mat = sio.loadmat(os.path.join(gt_path, 'wider_face_val.mat'))

    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    # Check first event (Parade)
    event_idx = 0
    event_name = str(event_list[event_idx][0][0])
    img_list = file_list[event_idx][0]

    print(f"\n{'='*70}")
    print(f"Event: {event_name}")
    print(f"{'='*70}")

    # Check if event exists in predictions
    if event_name in pred:
        print(f"✅ Event '{event_name}' found in predictions")
        pred_list = pred[event_name]
        print(f"   Predictions count: {len(pred_list)}")

        # Show first 5 prediction keys
        print(f"\n📋 First 5 prediction keys:")
        for i, key in enumerate(list(pred_list.keys())[:5]):
            print(f"   {i}: '{key}'")
    else:
        print(f"❌ Event '{event_name}' NOT found in predictions")
        print(f"   Available events: {list(pred.keys())}")

    # Check first 10 ground truth images
    print(f"\n📋 First 10 ground truth image names:")
    for j in range(min(10, len(img_list))):
        img_name = str(img_list[j][0][0])
        print(f"   {j}: '{img_name}'", end="")

        # Check if this image exists in predictions
        if event_name in pred:
            if img_name in pred[event_name]:
                print(f" ✅ FOUND (boxes: {len(pred[event_name][img_name])})")
            else:
                print(f" ❌ NOT FOUND")
                # Try to find similar keys
                similar = [k for k in pred[event_name].keys() if img_name in k or k in img_name]
                if similar:
                    print(f"      Similar keys: {similar[:3]}")
        else:
            print(f" ❌ Event not in predictions")

    # Check the problematic image
    problematic_img = '0_Parade_marchingband_1_465'
    print(f"\n{'='*70}")
    print(f"Checking problematic image: '{problematic_img}'")
    print(f"{'='*70}")

    if event_name in pred:
        if problematic_img in pred[event_name]:
            print(f"✅ FOUND in predictions!")
            print(f"   Boxes: {len(pred[event_name][problematic_img])}")
        else:
            print(f"❌ NOT FOUND in predictions")

            # Search for similar keys
            print(f"\n🔍 Searching for similar keys...")
            all_keys = list(pred[event_name].keys())

            # Search by substring
            matching = [k for k in all_keys if '465' in k]
            if matching:
                print(f"   Keys containing '465': {matching}")

            matching = [k for k in all_keys if 'marchingband' in k]
            if matching:
                print(f"   Keys containing 'marchingband': {matching[:5]}")

            # Show all keys if event is small
            if len(all_keys) <= 20:
                print(f"\n   All keys in '{event_name}':")
                for k in sorted(all_keys):
                    print(f"      '{k}'")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total events in predictions: {len(pred)}")
    print(f"Total events in ground truth: {len(event_list)}")

    for i in range(min(3, len(event_list))):
        evt_name = str(event_list[i][0][0])
        evt_imgs = len(file_list[i][0])

        if evt_name in pred:
            pred_imgs = len(pred[evt_name])
            print(f"\n  Event '{evt_name}':")
            print(f"    Ground truth images: {evt_imgs}")
            print(f"    Prediction files: {pred_imgs}")

            if pred_imgs != evt_imgs:
                print(f"    ⚠️  MISMATCH: {abs(evt_imgs - pred_imgs)} difference")
        else:
            print(f"\n  Event '{evt_name}': ❌ NOT IN PREDICTIONS")

if __name__ == '__main__':
    main()
