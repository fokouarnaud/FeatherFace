"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
# from bbox import bbox_overlaps
from IPython import embed

import numpy as np

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # split by literal \n in the file (corrupted format from test_widerface.py)
    # content looks like: "image_name\n11\n157 401 ... box data...\n..."
    # but with literal \n characters instead of real newlines
    parts = content.split('\\n')
    
    if len(parts) < 3:
        # fallback for correctly formatted files
        lines = content.split('\n')
        img_file = lines[0].strip() if lines else ''
        bbox_lines = lines[2:] if len(lines) > 2 else []
    else:
        # handle corrupted format
        img_file = parts[0].strip()
        num_boxes_str = parts[1].strip()
        # remaining parts are bbox data (5 values per box)
        bbox_lines = []
        for i in range(2, len(parts)):
            part = parts[i].strip()
            if part and not part.isdigit():  # skip the count number
                bbox_lines.append(part)
    
    # normalize image id to basename without extension for consistent keys
    img_basename = os.path.basename(img_file)
    img_name_noext = os.path.splitext(img_basename)[0]
    
    # parse bbox lines
    try:
        boxes = np.array(list(map(lambda x: [float(a) for a in x.strip().split()], bbox_lines))).astype('float')
    except (ValueError, IndexError):
        # handle lines that don't parse as floats
        boxes = np.zeros((0, 5), dtype='float')
    
    return img_name_noext, boxes




def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)

        # Skip if not a directory
        if not os.path.isdir(event_dir):
            continue

        print("event_dir")
        print(event_dir)

        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            # Skip if not a file
            img_path = os.path.join(event_dir, imgtxt)
            if not os.path.isfile(img_path):
                continue

            imgname, _boxes = read_pred_file(img_path)
            # imgname from read_pred_file is already basename without extension
            current_event[imgname] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh=0.5, debug=False):
    # keep original pred path for filesystem checks
    pred_path = pred
    preds = get_preds(pred_path)
    # normalize scores across all loaded predictions
    norm_score(preds)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)

    if debug:
        print(f"[DEBUG] pred_path = {pred_path}")
        print(f"[DEBUG] loaded events from predictions: {len(preds)}")
        sample_events = list(preds.keys())[:6]
        for ev in sample_events:
            try:
                print(f"  - {ev}: {len(preds[ev])} images (sample keys: {list(preds[ev].keys())[:6]})")
            except Exception:
                print(f"  - {ev}: <error listing keys>")
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            # safe lookup of event predictions
            pred_list = preds.get(event_name)
            if pred_list is None:
                # print helpful debug info and skip this event
                print(f"[WARN] No predictions found for event '{event_name}'.")
                # check filesystem for event folder
                event_dir = os.path.join(pred_path, event_name)
                if os.path.exists(event_dir):
                    files = os.listdir(event_dir)
                    print(f"  Found {len(files)} files in {event_dir}; sample: {files[:5]}")
                else:
                    print(f"  Event directory does not exist on disk: {event_dir}")
                continue
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                img_key = str(img_list[j][0][0])
                # image keys in preds are stored without extensions (basename)
                # try exact lookup first, then fallback to basename without extension
                pred_info = pred_list.get(img_key)
                if pred_info is None:
                    # fallback to basename without extension
                    img_key_base = os.path.splitext(os.path.basename(img_key))[0]
                    pred_info = pred_list.get(img_key_base)
                if pred_info is None:
                    # detailed debug output to help trace mismatch
                    #print(f"[DEBUG] Missing prediction for image '{img_key}' in event '{event_name}'")
                    sample_keys = list(pred_list.keys())[:10]
                    #print(f"  Available keys for event '{event_name}' (sample {len(sample_keys)}): {sample_keys}")
                    # Also print candidate path on disk
                    candidate_file = os.path.join(pred_path, event_name, img_key + '.txt')
                    candidate_file2 = os.path.join(pred_path, event_name, img_key_base + '.txt')
                    #print(f"  Candidate files: {candidate_file} -> exists? {os.path.exists(candidate_file)}")
                    #print(f"               {candidate_file2} -> exists? {os.path.exists(candidate_file2)}")
                    # skip this image
                    continue

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)
    # Use tqdm.write when available so printed results are not hidden/overwritten by tqdm
    def safe_print(*args, **kwargs):
        try:
            # compose message similarly to print, then use tqdm.write which is safe with progress bars
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            msg = sep.join(str(a) for a in args) + end
            # tqdm.write expects no trailing newline
            tqdm.tqdm.write(msg.rstrip('\n'))
        except Exception:
            print(*args, **kwargs, flush=True)

    safe_print("==================== Results ====================")
    safe_print("Easy   Val AP: {}".format(aps[0]))
    safe_print("Medium Val AP: {}".format(aps[1]))
    safe_print("Hard   Val AP: {}".format(aps[2]))
    safe_print("=================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred')
    parser.add_argument('-g', '--gt', default='/Users/Vic/Downloads/eval_tools/ground_truth/')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints showing loaded prediction keys')

    args = parser.parse_args()
    evaluation(args.pred, args.gt, debug=args.debug)