// services/logic-cpp/src/zmq_server.cpp

#include <iostream>
#include <zmq.hpp>
#include "entt_registry.hpp"
#include "calibration.hpp"

// 방금 도커에서 생성한 FlatBuffers 해독기 헤더 포함
#include "../schema/vision_data_generated.h"

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "[Logic Core] C++ 컨트롤 타워 부팅 중..." << std::endl;
    std::cout << "======================================" << std::endl;

    zmq::context_t context(1);
    InteriorRegistry ecs;
    SpaceCalibrator calibrator;

    zmq::socket_t sock_unity(context, zmq::socket_type::rep);
    sock_unity.bind("tcp://*:5555"); 

    zmq::socket_t sock_perception(context, zmq::socket_type::pull);
    sock_perception.bind("tcp://*:5556"); 

    zmq::socket_t sock_reconstruct(context, zmq::socket_type::pull);
    sock_reconstruct.bind("tcp://*:5557"); 

    zmq::socket_t sock_matching(context, zmq::socket_type::pull);
    sock_matching.bind("tcp://*:5558"); 

    zmq::pollitem_t items[] = {
        { static_cast<void*>(sock_unity), 0, ZMQ_POLLIN, 0 },
        { static_cast<void*>(sock_perception), 0, ZMQ_POLLIN, 0 },
        { static_cast<void*>(sock_reconstruct), 0, ZMQ_POLLIN, 0 },
        { static_cast<void*>(sock_matching), 0, ZMQ_POLLIN, 0 }
    };

    std::cout << "[Logic Core] 모든 포트 바인딩 완료. 수신 대기 중..." << std::endl;

    while (true) {
        zmq::poll(items, 4, -1);

        // [AI Perception] 데이터 수신
        if (items[1].revents & ZMQ_POLLIN) {
            zmq::message_t msg;
            sock_perception.recv(msg, zmq::recv_flags::none);
            
            // 데이터가 진짜 우리가 만든 FlatBuffers가 맞는지 검사합니다.
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(msg.data()), msg.size());
            if (!InteriorPlatform::VerifyVisionMessageBuffer(verifier)) {
                std::cerr << "[오류/경고] FlatBuffers 형식이 아닌 쓰레기 데이터가 들어왔습니다. 무시합니다." << std::endl;
                continue; // 서버를 죽이지 않고 다음 데이터를 기다립니다.
            }
            
            // --- 바이너리 해독 (FlatBuffers 파싱) ---
            auto vision_msg = InteriorPlatform::GetVisionMessage(msg.data());
            
            std::cout << "\n[Perception] --- 새로운 비전 데이터 수신 ---" << std::endl;
            std::cout << "프레임 타임스탬프: " << vision_msg->timestamp() << std::endl;

            // 탐지된 객체(가구) 목록 출력
            if (vision_msg->objects()) {
                std::cout << "탐지된 객체 수: " << vision_msg->objects()->size() << "개" << std::endl;
                
                for (auto obj : *vision_msg->objects()) {
                    std::cout << " 🔹 가구: [" << obj->label()->c_str() << "]" 
                              << " | 확신도: " << (int)(obj->confidence() * 100) << "%"
                              << " | 3D 좌표(추정): X=" << obj->position_3d()->x() 
                              << ", Y=" << obj->position_3d()->y() 
                              << ", Z=" << obj->position_3d()->z() << std::endl;
                }
            } else {
                std::cout << "탐지된 가구가 없습니다." << std::endl;
            }
            std::cout << "------------------------------------------\n";
        }
        
        // (나머지 포트 수신 로직은 생략/동일)
    }

    return 0;
}