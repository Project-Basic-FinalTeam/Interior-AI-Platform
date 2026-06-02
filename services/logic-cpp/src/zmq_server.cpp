// 파일 위치: /InteriorPlatform_Workspace/services/logic-cpp/src/
// 파일 명: zmq_server.cpp

#include <iostream>
#include <zmq.hpp>
#include <string>
#include <vector>
#include <map>
#include "../schema/vision_data_generated.h"
#include "rag_client.hpp" 

struct ObjectMemory {
    std::string label;
    float conf = 0.0f, x_min = 0.0f, y_min = 0.0f, x_max = 0.0f, y_max = 0.0f;
    float pos_x = 0.0f, pos_y = 0.0f, pos_z = 0.0f;
};

int main() {
    try {
        zmq::context_t context(1);
        zmq::socket_t sock_perception(context, zmq::socket_type::req);
        sock_perception.connect("tcp://ai-perception:5556");

        zmq::socket_t sock_reconstruction(context, zmq::socket_type::req);
        sock_reconstruction.connect("tcp://ai-reconstruction:5557");

        zmq::socket_t sock_unity(context, zmq::socket_type::rep);
        sock_unity.bind("tcp://*:5555");

        std::map<int, ObjectMemory> global_object_memory;
        RagClient rag_client;

        std::cout << "======================================" << std::endl;
        std::cout << "[Logic Core] 지능형 오케스트레이터 가동 완료!" << std::endl;
        std::cout << "======================================" << std::endl;

        while (true) {
            zmq::message_t unity_req;
            if (!sock_unity.recv(unity_req, zmq::recv_flags::none)) continue;
            
            std::string req_str(static_cast<char*>(unity_req.data()), unity_req.size());

            // ===================================================================
            // 🔥 [RAG 분기] ASK_RAG 명령 처리 (RAG + 3DGS 통합 자동화)
            // ===================================================================
            if (req_str.rfind("ASK_RAG|", 0) == 0) {
                size_t first_pipe = req_str.find('|');
                size_t second_pipe = req_str.find('|', first_pipe + 1);
                
                int target_id = 0;
                std::string user_query = "";
                
                if (second_pipe != std::string::npos) {
                    target_id = std::stoi(req_str.substr(first_pipe + 1, second_pipe - first_pipe - 1));
                    user_query = req_str.substr(second_pipe + 1);
                }

                std::cout << "[Logic Core] 📩 RAG 추천 요청 수신! (ID: " << target_id << ")" << std::endl;

                if (global_object_memory.count(target_id)) {
                    auto& m = global_object_memory[target_id];
                    // 1. RAG 서버에 추천 요청
                    std::string ans = rag_client.GetRecommendation(user_query, target_id, m.label, m.conf, m.x_min, m.y_min, m.x_max, m.y_max, m.pos_x, m.pos_y, m.pos_z);
                    std::cout << "[Logic Core] 🔍 GPT 응답 수신 완료." << std::endl;

                    // 2. JSON에서 storage_uri 파싱 후 ai-reconstruction 으로 전송
                    std::string recommended_ply = "asset_unknown.ply"; 
                    std::string uri_key = "\"storage_uri\":\"";
                    size_t pos = ans.find(uri_key);
                    
                    if (pos != std::string::npos) {
                        size_t start = pos + uri_key.length();
                        size_t end = ans.find("\"", start);
                        std::string storage_uri = ans.substr(start, end - start);
                        
                        std::cout << "[Logic Core] 🖼️ 추천 가구 이미지 발견! : " << storage_uri << std::endl;
                        std::cout << "[Logic Core] 🚀 3DGS 엔진(TRELLIS)으로 실시간 3D 변환을 지시합니다..." << std::endl;
                        
                        // ID 앞에 rag_ 를 붙여서 기존 가구 파일명과 충돌하지 않게 만듭니다.
                        std::string rag_obj_id = "rag_" + std::to_string(target_id);
                        std::string recon_cmd = "{\"obj_id\": \"" + rag_obj_id + "\", \"image_url\": \"" + storage_uri + "\"}";
                        
                        zmq::message_t recon_req(recon_cmd.c_str(), recon_cmd.size());
                        sock_reconstruction.send(recon_req, zmq::send_flags::none);
                        
                        zmq::message_t recon_reply;
                        sock_reconstruction.recv(recon_reply, zmq::recv_flags::none);
                        
                        std::cout << "[Logic Core] 🏁 3D 모델(.ply) 생성 완료!" << std::endl;
                        recommended_ply = "furniture_" + rag_obj_id + ".ply";
                    }

                    // 3. JSON 문자열의 마지막 닫는 괄호 '}' 바로 앞에 recommended_ply 끼워넣기
                    size_t last_brace = ans.find_last_of('}');
                    if (last_brace != std::string::npos) {
                        std::string inject_str = ",\"recommended_ply\":\"" + recommended_ply + "\"";
                        ans.insert(last_brace, inject_str);
                    }

                    // 4. 완성된 JSON을 유니티로 최종 전송!
                    zmq::message_t reply(ans.c_str(), ans.size());
                    sock_unity.send(reply, zmq::send_flags::none);
                } else {
                    std::string err = "{\"status\":\"error\", \"answer\":\"객체 정보 없음\", \"recommended_ply\":\"asset_unknown.ply\"}";
                    zmq::message_t reply(err.c_str(), err.size());
                    sock_unity.send(reply, zmq::send_flags::none);
                }
                continue; 
            }

            // ===================================================================
            // 👇 [원본 로직 100%] SCAN_ROOM 파이프라인
            // ===================================================================
            std::cout << "[Logic Core] 📩 1. 스캔 명령 수신!" << std::endl;

            std::string command = "SCAN_ROOM";
            zmq::message_t py_req(command.c_str(), command.size());
            sock_perception.send(py_req, zmq::send_flags::none);
            
            zmq::message_t py_reply;
            if (sock_perception.recv(py_reply, zmq::recv_flags::none)) {
                auto vision_data = InteriorPlatform::GetVisionMessage(py_reply.data());
                
                int object_count = 0;
                if (vision_data->objects() != nullptr) {
                    object_count = vision_data->objects()->size();
                    
                    for(int i=0; i<object_count; ++i) {
                        auto obj = vision_data->objects()->Get(i);
                        ObjectMemory m;
                        if(obj->label()) m.label = obj->label()->str();
                        m.conf = obj->confidence();
                        if(obj->bbox()) { 
                            m.x_min=obj->bbox()->x_min(); 
                            m.y_min=obj->bbox()->y_min(); 
                            m.x_max=obj->bbox()->x_max(); 
                            m.y_max=obj->bbox()->y_max(); 
                        }
                        if(obj->position_3d()) { 
                            m.pos_x = obj->position_3d()->x(); 
                            m.pos_y = obj->position_3d()->y(); 
                            m.pos_z = obj->position_3d()->z(); 
                        }
                        global_object_memory[obj->id()] = m;
                    }
                }

                int estimated_time_sec = object_count * 50;
                std::string count_msg = "COUNT:" + std::to_string(object_count) + "|EST_SEC:" + std::to_string(estimated_time_sec);
                zmq::message_t reply_count(count_msg.c_str(), count_msg.size());
                sock_unity.send(reply_count, zmq::send_flags::none);

                if (object_count == 0) continue; 

                // ===================================================================
                // 🔥 [핵심 수정: 순서 복원] 파이썬이 파일을 다 만들 때까지 기다립니다!
                // ===================================================================
                
                // 1. 유니티가 보낸 "GIVE_ME_FINAL_DATA" 신호만 일단 받아둠
                zmq::message_t dummy_req;
                sock_unity.recv(dummy_req, zmq::recv_flags::none); 

                std::cout << "[Logic Core] ⏳ 3D 복원 돌입 (유니티는 모든 파일이 완성될 때까지 대기합니다)..." << std::endl;

                // 2. 파이썬을 시켜서 3D 파일(ply)을 모두 생성
                for (int i = 0; i < object_count; ++i) {
                    auto obj = vision_data->objects()->Get(i);
                    int obj_id = obj->id();
                    float x = obj->bbox() ? obj->bbox()->x_min() : 0.0f;
                    float y = obj->bbox() ? obj->bbox()->y_min() : 0.0f;
                    float w = obj->bbox() ? (obj->bbox()->x_max() - x) : 0.0f;
                    float h = obj->bbox() ? (obj->bbox()->y_max() - y) : 0.0f;
                    
                    std::string recon_cmd = "{\"image_path\": \"/app/models/test_room.jpg\", \"obj_id\": \"" + std::to_string(obj_id) + "\", \"bbox\": [" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(w) + ", " + std::to_string(h) + "]}";
                    zmq::message_t recon_req(recon_cmd.c_str(), recon_cmd.size());
                    sock_reconstruction.send(recon_req, zmq::send_flags::none);
                    
                    zmq::message_t recon_reply;
                    sock_reconstruction.recv(recon_reply, zmq::recv_flags::none);
                }
                std::cout << "[Logic Core] 🏁 3D 복원 파이프라인 완벽 종료! 파일 준비 완료." << std::endl;

                // 3. 파일이 서버에 전부 준비된 이 시점에! 유니티에게 데이터를 넘겨줍니다.
                sock_unity.send(py_reply, zmq::send_flags::none);
                std::cout << "[Logic Core] ✅ 유니티에게 최종 데이터를 전송했습니다. (다운로드 시작)" << std::endl;
            }
        }
    } catch (const zmq::error_t& e) { return 1; }
    return 0;
}