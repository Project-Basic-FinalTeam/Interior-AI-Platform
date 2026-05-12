using UnityEngine;
using System.Threading.Tasks;
using NetMQ;
using NetMQ.Sockets;
using Google.FlatBuffers;
using InteriorPlatform;

public class ZmqReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    public string serverUrl = "tcp://localhost:5555";

    [Header("Manager Reference")]
    public FurnitureManager furnitureManager;

    // 이 함수를 UI 버튼에 연결할 겁니다!
    public void RequestScanFromCPP()
    {
        Debug.Log("[ZMQ] C++ 서버에 스캔 명령 하달 중...");

        // 파이썬 AI가 추론하는 동안 유니티 화면이 멈추지 않게 비동기로 처리
        Task.Run(() => {
            AsyncIO.ForceDotNet.Force();
            
            // 구독(SUB)이 아니라 요청(REQ) 소켓 사용
            using (var reqSocket = new RequestSocket())
            {
                reqSocket.Connect(serverUrl);
                reqSocket.SendFrame("BTN_CLICKED"); // C++ 깨우기

                byte[] rawData;
                // AI 추론 시간을 고려해 10초 정도 넉넉히 기다려줌
                if (reqSocket.TryReceiveFrameBytes(System.TimeSpan.FromSeconds(10), out rawData))
                {
                    var bb = new ByteBuffer(rawData);
                    var visionMsg = VisionMessage.GetRootAsVisionMessage(bb);

                    Debug.Log($"[ZMQ] 📬 데이터 배송 완료! 가구 수: {visionMsg.ObjectsLength}");

                    for (int i = 0; i < visionMsg.ObjectsLength; i++)
                    {
                        var obj = visionMsg.Objects(i).Value;
                        float scaledX = obj.Position3d.Value.X * 0.01f;
                        float scaledZ = obj.Position3d.Value.Y * 0.01f;

                        furnitureManager._furnitureQueue.Enqueue(new FurnitureManager.FurnitureData {
                            id = i, 
                            label = obj.Label, 
                            position = new Vector3(scaledX, 0f, scaledZ)
                        });
                    }
                }
                else
                {
                    Debug.LogError("[ZMQ] 응답 초과. 파이썬 AI가 아직 자고 있거나 에러가 났습니다.");
                }
            }
            NetMQConfig.Cleanup();
        });
    }
}