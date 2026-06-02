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
    // 🔥 첫 번째 파라미터로 const std::string& user_query 추가!
    std::string GetRecommendation(const std::string& user_query, int target_id, const std::string& label, float conf, 
                                  float x_min, float y_min, float x_max, float y_max, 
                                  float pos_x, float pos_y, float pos_z) {
        
        json payload = {
            {"query", user_query}, // 🔥 하드코딩된 문자열 대신 유니티에서 넘어온 실제 질문(user_query)을 삽입!
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

        // ===================================================================
        // 🔥 [핵심 수정] cpr::VerifySsl(false) 옵션을 추가하여 HTTPS 에러를 무시하고 통과시킵니다.
        // ===================================================================
        cpr::Response r = cpr::Post(
            cpr::Url{rag_api_url},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{payload.dump()},
            cpr::VerifySsl(false) 
        );

        if (r.status_code == 200) {
            try {
                auto response_json = json::parse(r.text);
                if (response_json["status"] == "success") {
                    std::cout << "[RAG Client] ✅ GPT 응답 도착!\n";
                    
                    // 텍스트만 빼서 주지 말고, 파이썬이 준 JSON 전체(r.text)를 그대로 유니티로 넘깁니다!
                    // 그래야 유니티가 recommended_ply 키를 뽑아 쓸 수 있습니다.
                    return r.text; 
                }
            } catch (const std::exception& e) {
                std::cerr << "[RAG Client] JSON 파싱 에러: " << e.what() << "\n";
            }
        } else {
            std::cerr << "[RAG Client] 🚨 HTTP 요청 실패! 상태 코드: " << r.status_code << "\n";
            std::cerr << "응답 에러: " << r.text << "\n";
        }

        // 에러가 났을 때도 유니티가 뻗지 않도록 JSON 형태로 가짜 응답을 만들어 보냅니다.
        return "{\"status\":\"error\", \"answer\":\"가구 추천을 불러오는 데 실패했습니다.\", \"recommended_ply\":\"asset_unknown.ply\"}";
    }
};

#endif