cmake_minimum_required(VERSION 3.14)
project(logic_core)

set(CMAKE_CXX_STANDARD 17)

# 1. PkgConfig 모듈 로드 (리눅스에서 라이브러리 찾는 표준 도구)
find_package(PkgConfig REQUIRED)

# 2. PkgConfig를 통해 ZeroMQ(libzmq) 찾기
pkg_check_modules(ZMQ REQUIRED libzmq)

include(FetchContent)

# 3. ZeroMQ C++ 바인딩 (zmq.hpp) 가져오기
FetchContent_Declare(
  cppzmq
  GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
  GIT_TAG        v4.10.0
)
FetchContent_MakeAvailable(cppzmq)

# 4. EnTT 라이브러리 가져오기
FetchContent_Declare(
  entt
  GIT_REPOSITORY https://github.com/skypjack/entt.git
  GIT_TAG        v3.12.2
)
FetchContent_MakeAvailable(entt)

# 5. 실행 파일 생성
add_executable(logic_core src/zmq_server.cpp)

# 6. 라이브러리 및 인클루드 경로 연결
# ZMQ_LIBRARIES와 ZMQ_INCLUDE_DIRS는 위 pkg_check_modules에서 자동으로 찾아준 변수입니다.
target_link_libraries(logic_core PRIVATE cppzmq EnTT::EnTT flatbuffers ${ZMQ_LIBRARIES})
target_include_directories(logic_core PRIVATE ${ZMQ_INCLUDE_DIRS})