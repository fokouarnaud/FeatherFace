# CELLULE 21 CORRIGÉE - COPIEZ CE CODE DANS VOTRE NOTEBOOK
# Import evaluation utilities
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm

def detect_faces_v2(model, image_path, cfg, device, 
                    confidence_threshold=0.5, nms_threshold=0.4):
    """
    Detect faces using V2 model - SOLUTION DÉFINITIVE
    Cette version corrige TOUS les problèmes de device mismatch
    """
    # Load and preprocess image
    img_raw = cv2.imread(str(image_path))
    if img_raw is None:
        print(f"❌ Impossible de charger l'image: {image_path}")
        return None, None, None
    
    img = np.float32(img_raw)
    im_height, im_width = img.shape[:2]
    
    # ✅ SOLUTION: Tous les tenseurs créés sur le device correct DÈS LE DÉBUT
    scale = torch.tensor([im_width, im_height, im_width, im_height], 
                        dtype=torch.float32, device=device)
    scale_landm = torch.tensor([im_width, im_height] * 5, 
                              dtype=torch.float32, device=device)
    
    # Resize and normalize
    img_size = cfg['image_size']
    img = cv2.resize(img, (img_size, img_size))
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    # Generate priors
    priorbox = PriorBox(cfg, image_size=(img_size, img_size))
    priors = priorbox.forward().to(device)
    
    print(f"🔍 Debug info:")
    print(f"  Device utilisé: {device}")
    print(f"  Image shape: {img.shape}, device: {img.device}")
    print(f"  Priors shape: {priors.shape}, device: {priors.device}")
    print(f"  Scale device: {scale.device}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(img)
        
        # ✅ IMPORTANT: Vérifier le format des outputs
        if isinstance(outputs, tuple) and len(outputs) == 3:
            loc, conf, landms = outputs
        else:
            print(f"❌ Format d'output inattendu: {type(outputs)}")
            return None, None, None
    
    print(f"  Model outputs - loc: {loc.shape}, conf: {conf.shape}, landms: {landms.shape}")
    print(f"  Output devices - loc: {loc.device}, conf: {conf.device}")
    
    # Decode predictions avec gestion d'erreur
    try:
        # Decode boxes
        boxes = decode(loc.data.squeeze(0), priors, cfg['variance'])
        print(f"  Decoded boxes shape: {boxes.shape}, device: {boxes.device}")
        
        # ✅ SOLUTION: Multiplication avec tenseurs sur même device
        boxes = boxes * scale  # Les deux sont maintenant sur le même device
        boxes = boxes.cpu().numpy()
        
        # Decode scores
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        
        # Decode landmarks
        landms = decode_landm(landms.data.squeeze(0), priors, cfg['variance'])
        print(f"  Decoded landms shape: {landms.shape}, device: {landms.device}")
        
        # ✅ SOLUTION: Multiplication avec tenseurs sur même device
        landms = landms * scale_landm  # Les deux sont maintenant sur le même device
        landms = landms.cpu().numpy()
        
    except RuntimeError as e:
        print(f"❌ Erreur lors du decode: {e}")
        print("🔧 Vérification des devices:")
        print(f"  loc device: {loc.device}")
        print(f"  priors device: {priors.device}")
        print(f"  scale device: {scale.device}")
        return None, None, None
    
    # Filter by confidence
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    landms = landms[inds]
    
    print(f"✅ Filtrage: {len(inds)} détections avec conf > {confidence_threshold}")
    
    # Apply NMS
    if len(boxes) > 0:
        try:
            keep = py_cpu_nms(np.hstack((boxes, scores[:, np.newaxis])), nms_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            landms = landms[keep]
            print(f"✅ NMS: {len(keep)} détections finales")
        except Exception as e:
            print(f"❌ Erreur NMS: {e}")
            return None, None, None
    else:
        print("⚠️ Aucune détection après filtrage par confiance")
    
    return boxes, scores, landms

print("✅ Fonction detect_faces_v2 corrigée chargée!")
print("🔧 Cette version inclut:")
print("  - Création des tenseurs directement sur le bon device")
print("  - Debug informatif pour traquer les erreurs")
print("  - Gestion d'erreur robuste")
print("  - Vérification du format des outputs")