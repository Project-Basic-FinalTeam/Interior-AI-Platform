#include <iostream>
#include <zmq.hpp>
#include "../schema/vision_data_generated.h"

int main() {
    zmq::context_t context(1);

    // 1. Python AI용 소켓 (C++이 고객: REQ)
    zmq::socket_t sock_perception(context, ZMQ_REQ);
    sock_perception.connect("tcp://ai-perception:5556");

    // 2. Unity용 소켓 (C++이 상담원: REP)
    zmq::socket_t sock_unity(context, ZMQ_REP);
    sock_unity.bind("tcp://*:5555");

    std::cout << "======================================" << std::endl;
    std::cout << "[Logic Core] 컨트롤 타워 (완벽 수동 모드) 부팅 완료" << std::endl;
    std::cout << "======================================" << std::endl;

    while (true) {
        // [A] 유니티가 '스캔 버튼'을 누를 때까지 여기서 CPU 0%로 무한 대기!
        zmq::message_t unity_req;
        sock_unity.recv(unity_req, zmq::recv_flags::none);
        std::cout << "\n[Logic Core] 1. 유니티에서 스캔 명령 수신!" << std::endl;

        // [B] 파이썬 AI 찌르기
        std::string command = "SCAN_ROOM";
        zmq::message_t py_req(command.c_str(), command.size());
        sock_perception.send(py_req, zmq::send_flags::none);
        std::cout << "[Logic Core] 2. 파이썬 AI 스캔 지시 완료" << std::endl;

        // [C] 파이썬 결과 대기
        zmq::message_t py_reply;
        sock_perception.recv(py_reply, zmq::recv_flags::none);
        std::cout << "[Logic Core] 3. 파이썬 AI 결과물 획득 완료!" << std::endl;

        // [D] 유니티로 결과물 배송
        sock_unity.send(py_reply, zmq::send_flags::none);
        std::cout << "[Logic Core] 4. 유니티로 배송 완료 (대기 모드로 복귀)" << std::endl;
    }

    return 0;
}