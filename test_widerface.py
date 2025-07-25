#!/usr/bin/env python3
"""
FeatherFace WIDERFace Testing Script
Adapted for CBAM baseline model with attention architecture

Scientific evaluation on WIDERFace dataset for CBAM baseline:
- CBAM Baseline (488,664 parameters): 6 CBAM modules (3 backbone + 3 BiFPN)

Note: For ECA-CBAM hybrid testing, use test_eca_cbam.py

Usage:
    python test_widerface.py -m weights/cbam/featherface_cbam_final.pth --network cbam
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_cbam_paper_exact
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.featherface_cbam_exact import FeatherFaceCBAMExact
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import time


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='FeatherFace WIDERFace Test')
parser.add_argument('-m', '--trained_model', default='./weights/cbam/featherface_cbam_final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='cbam', choices=['cbam'], 
                    help='Network architecture: cbam (baseline only)')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='WIDERFace', type=str, choices=['WIDERFace'], help='dataset')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    """Check model keys match pretrained keys"""
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """Remove module prefix from state dict keys"""
    print('remove prefix \\'{}\\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    """Load model weights safely"""
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    
    # Filter out thop profiling keys if present
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if not (k.endswith('total_ops') or k.endswith('total_params')):
            filtered_dict[k] = v
    
    check_keys(model, filtered_dict)
    model.load_state_dict(filtered_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # Use CBAM configuration
    cfg = cfg_cbam_paper_exact
    print("🔬 Testing CBAM Baseline (488,664 parameters)")
    print("   Architecture: 6 CBAM modules (3 backbone + 3 BiFPN)")
    print("   Scientific foundation: Woo et al. ECCV 2018")
    print("   Note: For ECA-CBAM hybrid testing, use test_eca_cbam.py")

    # net and model
    net = FeatherFaceCBAMExact(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    
    # Verify model parameter count
    total_params = sum(p.numel() for p in net.parameters())
    expected_params = 488664
    print(f'✓ Model loaded: {total_params:,} parameters (expected: {expected_params:,})')
    if abs(total_params - expected_params) > 100:
        print(f'⚠️  Warning: Parameter count mismatch by {total_params - expected_params:+,}')

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, "r") as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if img_raw is None:
            print(f"Warning: Could not load image {image_path}")
            continue

        img = np.float32(img_raw)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        _t['misc'].toc()

        # save dets
        if args.dataset == "WIDERFace":
            save_name = args.save_folder + img_name[:-4] + ".txt"
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                bboxs = dets
                file_name = os.path.basename(save_name)[:-4] + "\\n"
                bboxs_num = str(len(bboxs)) + "\\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \\n"
                    fd.write(line)

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

    # Print final statistics
    print(f'\\n🎯 CBAM Baseline Testing Complete!')
    print(f'   Model: {args.trained_model}')
    print(f'   Parameters: {total_params:,}')
    print(f'   Attention modules: 6 (CBAM: 3 backbone + 3 BiFPN)')
    print(f'   Images processed: {num_images}')
    print(f'   Average inference time: {_t["forward_pass"].average_time:.4f}s')
    print(f'   Results saved to: {args.save_folder}')
    print(f'\\n📊 Next step: Run evaluation with:')
    print(f'   python evaluate_widerface.py --model {args.trained_model} --network cbam --show_results')
    print(f'\\n🔗 For ECA-CBAM hybrid testing:')
    print(f'   python test_eca_cbam.py -m weights/eca_cbam/featherface_eca_cbam_final.pth --network eca_cbam')