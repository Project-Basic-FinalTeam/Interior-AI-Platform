// Assets/Scripts/Network/ZmqReceiver.cs
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

    // 🔥 중복 클릭 방지용 자물쇠
    private bool _isScanning = false; 

    public void RequestScanFromCPP()
    {
        // 자물쇠가 잠겨있으면 (이미 스캔 중이면) 무시!
        if (_isScanning)
        {
            Debug.LogWarning("[ZMQ] ⏳ 아직 이전 스캔 데이터를 처리 중입니다. 잠시만 기다려주세요!");
            return; 
        }

        Debug.Log("[ZMQ] 🚀 C++ 서버에 스캔 명령 하달 중...");
        _isScanning = true; // 스캔 시작! 자물쇠 잠금

        Task.Run(() => {
            AsyncIO.ForceDotNet.Force();
            using (var reqSocket = new RequestSocket())
            {
                reqSocket.Connect(serverUrl);
                reqSocket.SendFrame("BTN_CLICKED");

                byte[] rawData;
                
                // 🔥 대용량 3D 파일 복사 시간을 고려해 대기 시간을 10초 -> 30초로 넉넉하게 늘림!
                if (reqSocket.TryReceiveFrameBytes(System.TimeSpan.FromSeconds(30), out rawData))
                {
                    var bb = new ByteBuffer(rawData);
                    var visionMsg = VisionMessage.GetRootAsVisionMessage(bb);

                    Debug.Log($"[ZMQ] 📬 데이터 배송 완료! 가구 수: {visionMsg.ObjectsLength}");

                    if (visionMsg.Hands.HasValue)
                    {
                        bool isPinching = visionMsg.Hands.Value.IsPinching;
                        if (isPinching)
                        {
                            Debug.Log("🤏 [UI 연동] Pinch (꼬집기) 감지! 가구를 집어 들 준비 완료!");
                        }
                        else
                        {
                            Debug.Log("🖐️ [UI 연동] 손바닥 폄. 아무것도 잡고 있지 않습니다.");
                        }
                    }

                    for (int i = 0; i < visionMsg.ObjectsLength; i++)
                    {
                        var obj = visionMsg.Objects(i).Value;
                        float scaledX = obj.Position3d.Value.X * 0.01f;
                        float scaledZ = obj.Position3d.Value.Y * 0.01f;

                        string rawLabel = obj.Label;
                        string[] parts = rawLabel.Split('|');
                        string className = parts[0];
                        string plyFileName = parts.Length > 1 ? parts[1] : "";

                        furnitureManager._furnitureQueue.Enqueue(new FurnitureManager.FurnitureData {
                            id = i, 
                            label = className, 
                            plyPath = plyFileName, 
                            position = new Vector3(scaledX, 0f, scaledZ)
                        });
                    }
                }
                else
                {
                    Debug.LogError("[ZMQ] ❌ 응답 초과. 처리 시간이 30초를 넘었거나 서버에 문제가 있습니다.");
                }
            }
            NetMQConfig.Cleanup();
            
            // 모든 작업이 끝났으므로 자물쇠 해제!
            _isScanning = false; 
        });
    }
}