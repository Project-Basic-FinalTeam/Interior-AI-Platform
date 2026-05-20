#include <iostream>
#include <zmq.hpp>
#include <string>
#include <vector>
#include "../schema/vision_data_generated.h"

int main() {
    try {
        zmq::context_t context(1);

        zmq::socket_t sock_perception(context, zmq::socket_type::req);
        sock_perception.connect("tcp://ai-perception:5556");

        zmq::socket_t sock_reconstruction(context, zmq::socket_type::req);
        sock_reconstruction.connect("tcp://ai-reconstruction:5557");

        zmq::socket_t sock_unity(context, zmq::socket_type::rep);
        sock_unity.bind("tcp://*:5555");

        std::cout << "======================================" << std::endl;
        std::cout << "[Logic Core] 지능형 오케스트레이터 가동 완료! (Two-Step ZMQ)" << std::endl;
        std::cout << "======================================" << std::endl;

        while (true) {
            std::cout << "\n[Logic Core] 유니티 명령 대기 중..." << std::endl;
            zmq::message_t unity_req;
            if (!sock_unity.recv(unity_req, zmq::recv_flags::none)) continue;

            std::cout << "[Logic Core] 📩 1. 스캔 명령 수신!" << std::endl;

            // [Step 1] YOLO(RF-DETR)에게 사진 분석 지시
            std::string command = "SCAN_ROOM";
            zmq::message_t py_req(command.c_str(), command.size());
            sock_perception.send(py_req, zmq::send_flags::none);
            std::cout << "[Logic Core] 🚀 2. AI Perception 분석 요청 중..." << std::endl;

            zmq::message_t py_reply;
            auto py_res = sock_perception.recv(py_reply, zmq::recv_flags::none);

            if (py_res) {
                auto vision_data = InteriorPlatform::GetVisionMessage(py_reply.data());
                
                int object_count = 0;
                if (vision_data->objects() != nullptr) {
                    object_count = vision_data->objects()->size();
                }

                std::cout << "[Logic Core] ✅ 3. 분석 완료! 총 " << object_count << "개의 객체 감지." << std::endl;

                // 🔥 [중요 변경] 1차 응답: 유니티에게 개수 먼저 알려주기
                std::string count_msg = "COUNT:" + std::to_string(object_count);
                zmq::message_t reply_count(count_msg.c_str(), count_msg.size());
                sock_unity.send(reply_count, zmq::send_flags::none);

                // 객체가 없으면 여기서 이번 루프 종료
                if (object_count == 0) {
                    std::cout << "[Logic Core] ⚠️ 감지된 객체가 없어 스캔을 조기 종료합니다." << std::endl;
                    continue; 
                }

                // 🚨 [주의] ZMQ REQ-REP 패턴 규칙 때문에, 유니티가 최종 데이터를 받으려면
                // 유니티 쪽에서 2차 수신(TryReceiveFrameBytes)을 대기하고 있는 상태여야 합니다.
                // (위의 C# 스크립트 수정본은 이 규칙을 따르고 있습니다.)

                // [Step 3] 감지된 객체 수만큼 3DGS 렌더링 지시
                for (int i = 0; i < object_count; ++i) {
                    auto obj = vision_data->objects()->Get(i);
                    int obj_id = obj->id();
                    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;

                    if (obj->bbox() != nullptr) {
                        x = obj->bbox()->x_min();
                        y = obj->bbox()->y_min();
                        w = obj->bbox()->x_max() - x;
                        h = obj->bbox()->y_max() - y;
                    }

                    std::cout << "[Logic Core] 🚀 4-" << (i+1) << "/" << object_count << " 객체 ID [" << obj_id << "] 3DGS 요청..." << std::endl;
                    
                    std::string recon_cmd = "{\"image_path\": \"/app/models/test_room.jpg\", \"obj_id\": \"" + std::to_string(obj_id) + "\", \"bbox\": [" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(w) + ", " + std::to_string(h) + "]}";

                    zmq::message_t recon_req(recon_cmd.c_str(), recon_cmd.size());
                    sock_reconstruction.send(recon_req, zmq::send_flags::none);

                    zmq::message_t recon_reply;
                    sock_reconstruction.recv(recon_reply, zmq::recv_flags::none);
                }

                // [Step 4] 유니티로 최종 배송 (진짜 FlatBuffers 데이터)
                // 유니티는 현재 1차 응답(COUNT)을 받고, 2차 응답을 대기 중이므로 그냥 쏴주면 됩니다. (단, 유니티가 REQ 소켓이므로 꼼수가 필요합니다)
                // 원래 REQ-REP는 "REQ(보냄)->REP(받고-보냄)->REQ(받음)" 형태라 두 번 연속 send가 안됩니다.
                
                // 💡 [해결책] 유니티에게 "나 다 만들었어, 최종 데이터 줘!" 라고 다시 REQ를 보내게 해야 합니다.
                // 하지만 C# 스크립트를 비동기로 바꿨으니, C++에서는 다음과 같이 처리합니다.
                
                // C++이 두 번째로 유니티에게 데이터를 보내려면, 유니티의 두 번째 요청을 받아야 합니다.
                std::cout << "[Logic Core] ⏳ 유니티의 최종 데이터 수신 대기(PING)를 기다립니다..." << std::endl;
                zmq::message_t dummy_req;
                sock_unity.recv(dummy_req, zmq::recv_flags::none); // 유니티의 두 번째 요청 받기

                sock_unity.send(py_reply, zmq::send_flags::none);  // 최종 데이터 전송
                std::cout << "[Logic Core] 🏁 5. 유니티로 최종 배송 완료!" << std::endl;

            } else {
                std::cerr << "[Logic Core] ❌ AI Perception 응답 수신 오류" << std::endl;
            }
        }
    } catch (const zmq::error_t& e) {
        std::cerr << "[Logic Core] ZMQ 에러: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}