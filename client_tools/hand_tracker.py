# 파일 위치: client_tools/
# 파일 명: hand_tracker.py

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import cv2
import mediapipe as mp
import socket

# 🎛️ 좌표 증폭 배율 설정 (이 값을 키우면 손을 조금만 움직여도 화면 끝까지 닿습니다)
SCALE_X = 9.5   # 기존 5에서 대폭 상향 (양옆 가구 집기용)
SCALE_Y = 7.0   # 기존 5에서 상향 (위아래 범위 확장)
SCALE_Z = 20.0  # 앞뒤 이동 감도

# 통신 설정
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

# MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

print("웹캠 구동 시작... (q를 누르면 종료)")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1) # 거울 모드
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # 화면에 뼈대 그리기
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            thumb = handLms.landmark[4]
            index = handLms.landmark[8]

            # 설정된 배율(SCALE)을 적용하여 유니티 좌표 확장
            t_x = (thumb.x - 0.5) * SCALE_X
            t_y = -(thumb.y - 0.5) * SCALE_Y
            t_z = thumb.z * SCALE_Z
            
            i_x = (index.x - 0.5) * SCALE_X
            i_y = -(index.y - 0.5) * SCALE_Y
            i_z = index.z * SCALE_Z

            data_string = f"{t_x},{t_y},{t_z}|{i_x},{i_y},{i_z}"
            sock.sendto(str.encode(data_string), serverAddressPort)

    cv2.imshow("Hand Tracking (Python) - 뼈대가 보여야 정상입니다!", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()