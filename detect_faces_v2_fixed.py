#!/usr/bin/env python3
"""
Fixed detect_faces_v2 function with proper device handling
Solution for device mismatch in notebook
"""

import torch
import cv2
import numpy as np
from pathlib import Path

def detect_faces_v2(model, image_path, cfg, device, 
                    confidence_threshold=0.5, nms_threshold=0.4):
    """
    Detect faces using V2 model - FIXED VERSION
    
    This version properly handles device placement to avoid:
    RuntimeError: Expected all tensors to be on the same device
    """
    from layers.functions.prior_box import PriorBox
    from utils.nms.py_cpu_nms import py_cpu_nms
    from utils.box_utils import decode, decode_landm
    
    # Load and preprocess image
    img_raw = cv2.imread(str(image_path))
    if img_raw is None:
        return None, None, None
    
    img = np.float32(img_raw)
    im_height, im_width = img.shape[:2]
    
    # ✅ FIX: Create scale tensor on correct device
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    
    # Resize and normalize
    img_size = cfg['image_size']
    img = cv2.resize(img, (img_size, img_size))
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    # Generate priors
    priorbox = PriorBox(cfg, image_size=(img_size, img_size))
    priors = priorbox.forward().to(device)
    
    # Forward pass
    with torch.no_grad():
        loc, conf, landms = model(img)
    
    # Decode predictions
    boxes = decode(loc.data.squeeze(0), priors, cfg['variance'])
    boxes = boxes * scale  # ✅ Now both tensors are on same device
    boxes = boxes.cpu().numpy()
    
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
    landms = decode_landm(landms.data.squeeze(0), priors, cfg['variance'])
    # ✅ FIX: Create scale_landm tensor on correct device  
    scale_landm = torch.Tensor([im_width, im_height] * 5).to(device)
    landms = landms * scale_landm  # ✅ Now both tensors are on same device
    landms = landms.cpu().numpy()
    
    # Filter by confidence
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    landms = landms[inds]
    
    # Apply NMS
    if len(boxes) > 0:
        keep = py_cpu_nms(np.hstack((boxes, scores[:, np.newaxis])), nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        landms = landms[keep]
    
    return boxes, scores, landms

def test_detection_function():
    """Test the fixed detection function"""
    print("Testing fixed detect_faces_v2 function...")
    
    # This is the corrected code for the notebook
    notebook_fix = """
# CORRECTED detect_faces_v2 function for notebook
def detect_faces_v2(model, image_path, cfg, device, 
                    confidence_threshold=0.5, nms_threshold=0.4):
    \"\"\"Detect faces using V2 model - DEVICE FIXED\"\"\"
    # Load and preprocess image
    img_raw = cv2.imread(str(image_path))
    if img_raw is None:
        return None, None, None
    
    img = np.float32(img_raw)
    im_height, im_width = img.shape[:2]
    
    # ✅ FIXED: Create tensors on correct device
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    
    # Resize and normalize
    img_size = cfg['image_size']
    img = cv2.resize(img, (img_size, img_size))
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    # Generate priors
    priorbox = PriorBox(cfg, image_size=(img_size, img_size))
    priors = priorbox.forward().to(device)
    
    # Forward pass
    with torch.no_grad():
        loc, conf, landms = model(img)
    
    # Decode predictions
    boxes = decode(loc.data.squeeze(0), priors, cfg['variance'])
    boxes = boxes * scale  # ✅ Both tensors now on same device
    boxes = boxes.cpu().numpy()
    
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
    landms = decode_landm(landms.data.squeeze(0), priors, cfg['variance'])
    # ✅ FIXED: Create scale_landm on correct device
    scale_landm = torch.Tensor([im_width, im_height] * 5).to(device)
    landms = landms * scale_landm  # ✅ Both tensors now on same device  
    landms = landms.cpu().numpy()
    
    # Filter by confidence
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds] 
    landms = landms[inds]
    
    # Apply NMS
    if len(boxes) > 0:
        keep = py_cpu_nms(np.hstack((boxes, scores[:, np.newaxis])), nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        landms = landms[keep]
    
    return boxes, scores, landms
    """
    
    print("Copy this corrected function to your notebook cell 21")
    return notebook_fix

if __name__ == "__main__":
    test_detection_function()