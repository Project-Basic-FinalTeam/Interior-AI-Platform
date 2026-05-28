// 파일 위치: /InteriorPlatform_Workspace/services/logic-cpp/src/
// 파일 명: rag_client.hpp

#ifndef RAG_CLIENT_HPP
#define RAG_CLIENT_HPP

#include <iostream>
#include <string>
#include <cpr/cpr.h>          // HTTP 통신용 (Python의 requests와 유사)
#include <nlohmann/json.hpp>  // JSON 파싱용

using json = nlohmann::json;

class RagClient {
private:
    std::string rag_api_url = "https://interplacental-liana-puddly.ngrok-free.dev/rag/perception-query";

public:
    // C++ EnTT 레지스트리에서 뽑아낸 데이터를 매개변수로 받습니다.
    std::string GetRecommendation(int target_id, const std::string& label, float conf, 
                                  float x_min, float y_min, float x_max, float y_max, 
                                  float pos_x, float pos_y, float pos_z) {
        
        // 1. RAG 서버가 요구하는 정확한 규격으로 JSON 페이로드 조립
        json payload = {
            {"query", "감지된 객체 자리에 들어갈 수 있는 상품을 추천해줘"},
            {"target_object_id", target_id},
            {"clearance_cm", 2},
            {"allow_xy_rotation", true},
            {"limit", 100},
            {"perception_objects", json::array({
                {
                    {"id", target_id},
                    {"label", label},
                    {"confidence", conf},
                    {"bbox", {
                        {"x_min", x_min}, {"y_min", y_min}, {"x_max", x_max}, {"y_max", y_max}
                    }},
                    {"position_3d", {
                        {"x", pos_x}, {"y", pos_y}, {"z", pos_z}
                    }}
                }
            })}
        };

        std::cout << "[RAG Client] 🚀 GPT 추론 서버로 분석 요청 중... (ID: " << target_id << ")\n";

        // 2. RAG API로 HTTP POST 요청 발사
        cpr::Response r = cpr::Post(
            cpr::Url{rag_api_url},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{payload.dump()}
        );

        // 3. 응답 처리 및 파싱
        if (r.status_code == 200) {
            try {
                auto response_json = json::parse(r.text);
                if (response_json["status"] == "success") {
                    std::string answer = response_json["answer"];
                    std::cout << "[RAG Client] ✅ GPT 응답 도착: " << answer << "\n";
                    
                    // 유니티로 다시 보내기 위해 GPT의 최종 답변(answer)만 반환합니다.
                    // 필요하다면 storage_uri도 함께 파싱해서 넘길 수 있습니다.
                    return answer; 
                }
            } catch (const std::exception& e) {
                std::cerr << "[RAG Client] JSON 파싱 에러: " << e.what() << "\n";
            }
        } else {
            std::cerr << "[RAG Client] 🚨 HTTP 요청 실패! 상태 코드: " << r.status_code << "\n";
            std::cerr << "응답 에러: " << r.text << "\n";
        }

        return "가구 추천을 불러오는 데 실패했습니다.";
    }
};

#endif