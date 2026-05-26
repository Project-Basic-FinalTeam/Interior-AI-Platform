import os
import torch
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase

class RFDETRHandler:
    def __init__(self, model_path: str = None, device: str = None, confidence: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence = confidence
        self.model_path = model_path
        self.model = None
        self.class_names = {}
        # IoU 임계값 (겹침을 판단하는 기준, 0.3 이하는 다른 객체로 간주)
        self.iou_threshold = 0.3

        print(f"[RF-DETR 모듈] Initializing (device={self.device})")
        self._load_model()

    def _load_model(self):
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = RFDETRBase(pretrain_weights=self.model_path)
                print(f"[RF-DETR 모듈] Custom weights loaded: {self.model_path}")
            else:
                self.model = RFDETRBase()
                print("[RF-DETR 모듈] Default weights loaded")

            if hasattr(self.model, "class_names"):
                self.class_names = self.model.class_names
        except Exception as e:
            print(f"[RF-DETR 모듈] ⚠️ Model load failed: {e}")
            self.model = None

    def _apply_nms(self, boxes, confs, iou_threshold):
        """IoU 기반 비최대 억제 (중복 객체 제거)"""
        if len(boxes) == 0: return []
        boxes = np.array(boxes)
        confs = np.array(confs)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = confs.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1: break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep

    def detect(self, image, threshold: float = None):
        if self.model is None: return []
        conf = threshold if threshold is not None else self.confidence
        pil_image = self._to_pil(image)

        try:
            detections = self.model.predict(pil_image, threshold=conf)
            return self._format_detections(detections)
        except Exception as e:
            print(f"[RF-DETR 모듈] Inference failed: {e}")
            return []

    def _to_pil(self, image) -> Image.Image:
        if isinstance(image, Image.Image): return image
        if isinstance(image, str): return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _format_detections(self, detections) -> list:
        results = []
        try:
            xyxy = getattr(detections, "xyxy", None)
            conf = getattr(detections, "confidence", None)
            class_id = getattr(detections, "class_id", None)

            if xyxy is None or len(xyxy) == 0: return results

            # 1. NMS 적용
            keep_indices = self._apply_nms(xyxy, conf, self.iou_threshold)
            
            # 2. 필터링된 결과만 추출
            for i in keep_indices:
                cid = int(class_id[i]) if class_id is not None else -1
                
                if isinstance(self.class_names, dict):
                    label = self.class_names.get(cid, str(cid))
                elif isinstance(self.class_names, (list, tuple)):
                    label = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
                else:
                    label = str(cid)

                results.append({
                    "bbox": [float(v) for v in xyxy[i]],
                    "label": label,
                    "confidence": float(conf[i]),
                    "class_id": cid,
                })
        except Exception as e:
            print(f"[RF-DETR 모듈] Result parsing failed: {e}")
        return results