import math

class DepthEstimator:
    def __init__(self, camera_height=1.2, fov_degrees=60.0, camera_pitch_degrees=15.0):
        self.camera_height = camera_height
        self.fov_degrees = fov_degrees
        # 🔥 스마트폰을 들고 방을 찍을 때의 평균적인 하향각(15도) 적용
        self.camera_pitch_degrees = camera_pitch_degrees 
        print(f"[Depth 모듈] 📐 디지털 트윈 공간 역산기 가동 (하향각: {camera_pitch_degrees}도)")

    def estimate_3d_position_and_scale(self, image_shape, bbox):
        img_h, img_w = image_shape[:2]
        x1, y1, x2, y2 = bbox

        cx = img_w / 2.0
        cy = img_h / 2.0
        bx = (x1 + x2) / 2.0
        by = (y1 + y2) / 2.0
        bw = x2 - x1  # 픽셀 너비
        bh = y2 - y1  # 픽셀 높이

        fov_rad = math.radians(self.fov_degrees)
        focal_length = (img_h / 2.0) / math.tan(fov_rad / 2.0)

        # [해결 1] 카메라가 아래를 내려다보고 있다고 가정하여 지평선(Horizon)을 화면 위쪽으로 올려줍니다.
        pitch_rad = math.radians(self.camera_pitch_degrees)
        horizon_y = cy - (focal_length * math.tan(pitch_rad))

        delta_y = y2 - horizon_y
        if delta_y < 10:
            delta_y = 10 

        # 깊이(Z) 계산 후 현실적인 실내 거리(0.5m ~ 7m)로 제한하여 우주로 날아가는 것 방지
        estimated_z = (self.camera_height * focal_length) / delta_y
        estimated_z = min(max(estimated_z, 0.5), 7.0)

        estimated_x = (bx - cx) * estimated_z / focal_length
        estimated_y = -(by - cy) * estimated_z / focal_length 

        # [해결 2] 깊이(Z)를 바탕으로 2D 픽셀이 현실에서 몇 미터인지 '물리적 스케일' 계산
        physical_w = (bw * estimated_z) / focal_length
        physical_h = (bh * estimated_z) / focal_length

        # 좌표(X,Y,Z)와 스케일(W,H)을 모두 반환
        return float(estimated_x), float(estimated_y), float(estimated_z), float(physical_w), float(physical_h)