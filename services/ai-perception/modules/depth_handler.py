# services/ai-perception/modules/depth_handler.py
import random

class DepthEstimator:
    def __init__(self):
        print("[Depth 모듈] 더미(Dummy) 깊이 추론기 초기화 완료 (다른 팀 개발 대기 중)")

    def estimate_depth(self, image, bbox):
        """
        주어진 이미지와 바운딩 박스 영역의 평균 깊이(Z축 거리)를 계산합니다.
        현재는 다른 팀의 개발이 완료될 때까지 하드코딩된 더미 값을 반환합니다.
        """
        x1, y1, x2, y2 = bbox
        
        # 유니티에서 가구들이 완전히 겹쳐 보이지 않도록 
        # 2.0m ~ 3.0m 사이의 임의의 거리를 던져줍니다. (나중에 실제 모델로 교체)
        dummy_z = 2.0 + random.uniform(0.0, 1.0)
        
        return dummy_z