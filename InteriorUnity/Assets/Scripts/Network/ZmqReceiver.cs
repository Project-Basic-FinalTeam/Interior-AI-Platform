using UnityEngine;
using System;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using NetMQ;
using NetMQ.Sockets;
using Google.FlatBuffers;
using InteriorPlatform;

public class ZmqReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    public string serverUrl; 

    [Header("Manager Reference")]
    public FurnitureManager furnitureManager;

    private bool _isScanning = false;
    
    // 백그라운드 스레드에서 수신한 원시 바이트 데이터를 메인 스레드로 안전하게 전달하기 위한 큐
    private ConcurrentQueue<byte[]> _mainThreadActionQueue = new ConcurrentQueue<byte[]>();

    void Awake()
    {
        AsyncIO.ForceDotNet.Force();
        
        // 인스펙터의 값을 무시하고 여기서 IP를 강제 주입합니다.
        serverUrl = "tcp://100.118.177.19:5555"; 
        Debug.Log($"[ZMQ] 통신 모듈 초기화 및 접속 주소 강제 고정 완료: {serverUrl}");
    }

    void OnApplicationQuit()
    {
        NetMQConfig.Cleanup(false);
        Debug.Log("[ZMQ] 통신 모듈 안전 종료 완료.");
    }

    // 매 프레임마다 백그라운드 스레드에서 전달된 3D 가구 위치 연산 작업이 있는지 확인
    void Update()
    {
        if (_mainThreadActionQueue.TryDequeue(out byte[] rawData))
        {
            // 메인 스레드 컨텍스트 내부이므로 UnityEngine API(Camera.main 등)를 안전하게 호출합니다.
            ProcessResultOnMainThread(rawData);
        }
    }

    public void RequestScanFromCPP()
    {
        if (_isScanning)
        {
            Debug.LogWarning("[ZMQ] ⏳ 아직 이전 스캔 데이터를 처리 중입니다.");
            return;
        }

        Debug.Log("[ZMQ] 🚀 C++ 서버에 스캔 명령 하달 중...");
        _isScanning = true;

        Task.Run(() => {
            try
            {
                using (var reqSocket = new RequestSocket())
                {
                    reqSocket.Options.Linger = TimeSpan.FromSeconds(1);
                    reqSocket.Connect(serverUrl);
                    
                    // [Step 1] 1차 스캔 명령 전송
                    byte[] commandBytes = System.Text.Encoding.UTF8.GetBytes("SCAN_ROOM");
                    reqSocket.SendFrame(commandBytes);  

                    string initialResponse = "";
                    bool gotFirstReply = reqSocket.TryReceiveFrameString(TimeSpan.FromSeconds(30), out initialResponse);

                    if (!gotFirstReply)
                    {
                        Debug.LogError("[ZMQ] ❌ 1차 응답(객체 개수) 수신 실패. 서버 연결을 확인하세요.");
                        return;
                    }

                    if (initialResponse.StartsWith("COUNT:"))
                    {
                        int objectCount = int.Parse(initialResponse.Split(':')[1]);
                        Debug.Log($"[ZMQ] 💡 C++ 서버 보고: 총 {objectCount}개의 객체를 감지했습니다. 3D 변환을 시작합니다...");

                        if (objectCount == 0)
                        {
                            Debug.LogWarning("[ZMQ] 감지된 객체가 없어 스캔을 조기 종료합니다.");
                            return;
                        }

                        int timeoutSeconds = Math.Max(30, objectCount * 10);
                        Debug.Log($"[ZMQ] ⏳ 예상 소요 시간 기반 타임아웃 설정: {timeoutSeconds}초 대기");

                        // [Step 2] 2차 응답을 수신하기 위해 핑(Ping) 전송
                        byte[] pingBytes = System.Text.Encoding.UTF8.GetBytes("GIVE_ME_FINAL_DATA");
                        reqSocket.SendFrame(pingBytes);

                        byte[] rawData = null;
                        bool gotFinalReply = reqSocket.TryReceiveFrameBytes(TimeSpan.FromSeconds(timeoutSeconds), out rawData);

                        if (gotFinalReply)
                        {
                            // 원시 바이트 데이터를 컨커런트 큐에 삽입하여 메인 스레드(Update)로 작업을 이관합니다.
                            _mainThreadActionQueue.Enqueue(rawData);
                        }
                        else
                        {
                            Debug.LogError($"[ZMQ] ❌ {timeoutSeconds}초 응답 초과. 너무 오래 걸렸거나 서버가 끊어졌습니다.");
                        }
                    }
                    else
                    {
                         Debug.LogError($"[ZMQ] ❌ 잘못된 1차 응답 형식: {initialResponse}");
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[ZMQ] 🚨 통신 중 치명적 에러 상세 로그:\n{e.ToString()}");
            }
            finally
            {
                _isScanning = false;
            }
        });
    }

    // 메인 스레드에서만 안전하게 호출되는 실제 인스턴스 소환 위치 처리 함수
    private void ProcessResultOnMainThread(byte[] rawData)
    {
        try
        {
            var bb = new ByteBuffer(rawData);
            var visionMsg = VisionMessage.GetRootAsVisionMessage(bb);
            Debug.Log($"[ZMQ] 📬 메인 스레드로 데이터 배송 완료! 가구 수: {visionMsg.ObjectsLength}");

            Transform camTransform = Camera.main.transform;

            for (int i = 0; i < visionMsg.ObjectsLength; i++)
            {
                var obj = visionMsg.Objects(i).Value;

                // 파이썬 엣지 서버가 갱신하여 패킹해 준 새로운 라벨 파싱
                string rawLabel = obj.Label;
                string[] parts = rawLabel.Split('|');
                string className = parts[0];
                string plyFileName = parts.Length > 1 ? parts[1] : "";
                
                // 🔥 [추가 항목] 문자열 파싱을 통해 물리 스케일 값(미터 단위)을 복원합니다.
                float scaleW = parts.Length > 2 ? float.Parse(parts[2]) : 1.0f;
                float scaleH = parts.Length > 3 ? float.Parse(parts[3]) : 1.0f;

                // 파이썬 수식 모듈(DepthEstimator)이 역산해 준 카메라 기준의 3D 미터 단위 상대 위치
                float localX = obj.Position3d.Value.X;
                float localY = obj.Position3d.Value.Y;
                float localZ = obj.Position3d.Value.Z;

                Vector3 localPosition = new Vector3(localX, localY, localZ);

                // TransformPoint를 이용하여 카메라의 회전(Rotation) 및 전방 방향(Forward) 행렬을 곱해 
                // 월드 좌표계 상의 최종 물리 위치를 정확하게 찾아냅니다.
                Vector3 finalWorldPosition = camTransform.TransformPoint(localPosition);

                // 🔥 [수정 항목] 스케일 벡터 데이터 조립 후 구조체 큐 인입
                furnitureManager._furnitureQueue.Enqueue(new FurnitureManager.FurnitureData {
                    id = i,
                    label = className,
                    plyPath = plyFileName,
                    position = finalWorldPosition, // 렌즈 기하학 기반 실물 매핑 공간
                    scale = new Vector3(scaleW, scaleH, scaleW) // 물리 크기 복원 데이터 매핑 (Z축은 두께 보정용 W 사용)
                });
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[ZMQ] 🚨 수신 데이터 분석 중 에러 발생:\n{ex.ToString()}");
        }
    }
}