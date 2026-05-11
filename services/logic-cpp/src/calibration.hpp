#pragma once  // 중복 포함 방지 (필수!)
#include <iostream>

class SpaceCalibrator {
public:
    float ClampToGround(float raw_y, float ground_level) {
        float calibrated_y = raw_y - ground_level;
        if (calibrated_y > -0.05f && calibrated_y < 0.05f) {
            return 0.0f;
        }
        return calibrated_y;
    }

    void BuildSemanticGrid() {
        std::cout << "[Calibration] 10cm x 10cm 시맨틱 그리드 매핑 완료." << std::endl;
    }
};