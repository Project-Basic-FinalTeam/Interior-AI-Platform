# services/ai-perception/modules/rfdetr_handler.py
import os
import torch
import numpy as np
from PIL import Image


class RFDETRDetector:
    """
    RF-DETR 기반 객체 탐지 래퍼 클래스.
    YOLO 대체 또는 보조 탐지기로 사용 가능합니다.
    """

    def __init__(self, model_path: str = None, device: str = None, confidence: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence = confidence
        self.model_path = model_path
        self.model = None
        self.class_names = {}

        print(f"[RF-DETR 모듈] 초기화 시작 (device={self.device})")
        self._load_model()

    def _load_model(self):
        """
        RF-DETR 모델을 로드합니다.
        rfdetr 패키지(`pip install rfdetr`)가 설치되어 있어야 합니다.
        """
        try:
            from rfdetr import RFDETRBase

            if self.model_path and os.path.exists(self.model_path):
                self.model = RFDETRBase(pretrain_weights=self.model_path)
                print(f"[RF-DETR 모듈] 커스텀 가중치 로드 완료: {self.model_path}")
            else:
                self.model = RFDETRBase()
                print("[RF-DETR 모듈] 기본 사전학습 가중치 로드 완료")

            if hasattr(self.model, "class_names"):
                self.class_names = self.model.class_names
        except ImportError:
            print("[RF-DETR 모듈] ⚠️  rfdetr 패키지가 설치되어 있지 않습니다. "
                  "`pip install rfdetr` 후 다시 시도하세요. (현재는 더미 모드로 동작)")
            self.model = None
        except Exception as e:
            print(f"[RF-DETR 모듈] ⚠️  모델 로드 실패: {e} (더미 모드로 동작)")
            self.model = None

    def predict(self, image, threshold: float = None):
        """
        이미지에서 객체를 탐지합니다.

        Args:
            image: numpy.ndarray (BGR, OpenCV 형식) 또는 PIL.Image 또는 파일 경로
            threshold: confidence 임계값 (None이면 self.confidence 사용)

        Returns:
            List[dict]: [{"bbox": [x1, y1, x2, y2], "label": str, "confidence": float, "class_id": int}, ...]
        """
        if self.model is None:
            return []

        conf = threshold if threshold is not None else self.confidence
        pil_image = self._to_pil(image)

        try:
            detections = self.model.predict(pil_image, threshold=conf)
        except Exception as e:
            print(f"[RF-DETR 모듈] 추론 실패: {e}")
            return []

        return self._format_detections(detections)

    def _to_pil(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            # OpenCV는 BGR, PIL은 RGB
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"지원하지 않는 이미지 타입: {type(image)}")

    def _format_detections(self, detections) -> list:
        """
        rfdetr 패키지의 출력(supervision.Detections 등)을 공통 dict 포맷으로 변환합니다.
        """
        results = []
        try:
            xyxy = getattr(detections, "xyxy", None)
            conf = getattr(detections, "confidence", None)
            class_id = getattr(detections, "class_id", None)

            if xyxy is None:
                return results

            for i in range(len(xyxy)):
                cid = int(class_id[i]) if class_id is not None else -1
                label = self.class_names.get(cid, str(cid)) if self.class_names else str(cid)
                results.append({
                    "bbox": [float(v) for v in xyxy[i]],
                    "label": label,
                    "confidence": float(conf[i]) if conf is not None else 0.0,
                    "class_id": cid,
                })
        except Exception as e:
            print(f"[RF-DETR 모듈] 결과 파싱 실패: {e}")

        return results
