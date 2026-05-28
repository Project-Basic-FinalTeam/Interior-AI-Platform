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

# =====================================================================
# 🔥 [최소 추가 구간] RAG 통신을 위한 JSON과 CPR 라이브러리만 딱 추가합니다.
# =====================================================================

# 5. nlohmann_json (JSON 파싱 라이브러리) 가져오기
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# 6. cpr (HTTP 통신 라이브러리) 가져오기
# (이전에 발생한 SSL/빌드 에러를 막기 위해, 무거운 테스트는 다 끄고 시스템 curl을 쓰도록 강제합니다)
set(CPR_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CPR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CPR_USE_SYSTEM_CURL ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  cpr
  GIT_REPOSITORY https://github.com/libcpr/cpr.git
  GIT_TAG        1.11.1
)
FetchContent_MakeAvailable(cpr)

# =====================================================================

# 7. 실행 파일 생성
add_executable(logic_core src/zmq_server.cpp)

# 8. 라이브러리 및 인클루드 경로 연결
# 🔥 원본 링킹 목록 맨 끝에 `cpr::cpr`과 `nlohmann_json::nlohmann_json`만 살포시 추가했습니다.
target_link_libraries(logic_core PRIVATE 
    cppzmq 
    EnTT::EnTT 
    flatbuffers 
    ${ZMQ_LIBRARIES} 
    cpr::cpr 
    nlohmann_json::nlohmann_json
)
target_include_directories(logic_core PRIVATE ${ZMQ_INCLUDE_DIRS})