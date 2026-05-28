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
    public string serverUrl = "tcp://100.118.177.19:5555"; 

    [Header("Manager Reference")]
    public FurnitureManager furnitureManager;

    private bool _isScanning = false;
    private ConcurrentQueue<byte[]> _mainThreadActionQueue = new ConcurrentQueue<byte[]>();

    void Awake()
    {
        // ❌ 여기에 있던 문제의 Cleanup 자폭 코드를 삭제했습니다!
        AsyncIO.ForceDotNet.Force();
        serverUrl = "tcp://100.118.177.19:5555".Trim(); 
        Debug.Log($"[ZMQ] 통신 모듈 초기화 완료: {serverUrl}");
    }

    void OnDestroy()
    {
        // 통신 종료는 프로그램이 꺼질 때만 실행해야 합니다.
        NetMQConfig.Cleanup(false);
        Debug.Log("[ZMQ] 통신 모듈 안전 종료.");
    }

    void Update()
    {
        if (_mainThreadActionQueue.TryDequeue(out byte[] rawData))
        {
            ProcessResultOnMainThread(rawData);
        }
    }

    public void RequestScanFromCPP()
    {
        if (_isScanning) return;
        _isScanning = true;

        Task.Run(() => {
            try
            {
                using (var reqSocket = new RequestSocket())
                {
                    reqSocket.Options.Linger = TimeSpan.FromSeconds(1); 
                    Debug.Log($"[ZMQ] {serverUrl} 로 연결 시도 중...");
                    reqSocket.Connect(serverUrl);
                    reqSocket.SendFrame(System.Text.Encoding.UTF8.GetBytes("SCAN_ROOM")); 
                    
                    // 1차 응답 수신 대기
                    if (reqSocket.TryReceiveFrameString(TimeSpan.FromSeconds(30), out string initialResponse) && initialResponse.StartsWith("COUNT:"))
                    {
                        // 🔥 수정 1: "COUNT:3|EST_SEC:90" 형태의 문자열 분리 및 파싱
                        string[] parts = initialResponse.Split('|');
                        int objectCount = int.Parse(parts[0].Replace("COUNT:", ""));
                        string estTime = parts.Length > 1 ? parts[1].Replace("EST_SEC:", "") : "알 수 없음";

                        // 🔥 수정 2: 유니티 콘솔에 기대했던 안내 메시지 출력
                        Debug.Log($"[3DGS] ✅ 스캔 완료! 총 {objectCount}개의 가구가 감지되었습니다. (렌더링 예상 대기 시간: 약 {estTime}초)");

                        if (objectCount > 0)
                        {
                            reqSocket.SendFrame(System.Text.Encoding.UTF8.GetBytes("GIVE_ME_FINAL_DATA"));
                            
                            // 🔥 수정 3: 4090이 3DGS를 다 깎을 때까지 유니티가 통신을 끊지 않고 기다리도록 타임아웃 연장 (예상 시간 + 10초 여유 버퍼)
                            int waitTimeout = parts.Length > 1 ? int.Parse(estTime) + 30 : Math.Max(60, objectCount * 50);
                            
                            if (reqSocket.TryReceiveFrameBytes(TimeSpan.FromSeconds(waitTimeout), out byte[] rawData))
                            {
                                _mainThreadActionQueue.Enqueue(rawData);
                            }
                            else 
                            {
                                Debug.LogError($"[ZMQ] 🚨 3DGS 렌더링 응답 시간 초과 (Timeout: {waitTimeout}초)");
                            }
                        }
                    }
                }
            }
            catch (Exception e) { 
                Debug.LogError($"[ZMQ] 통신 에러: {e.Message}"); 
            }
            finally { 
                _isScanning = false; 
            }
        });
    }

    private void ProcessResultOnMainThread(byte[] rawData)
    {
        try
        {
            var bb = new ByteBuffer(rawData);
            var visionMsg = VisionMessage.GetRootAsVisionMessage(bb);
            Transform camTransform = Camera.main.transform;

            for (int i = 0; i < visionMsg.ObjectsLength; i++)
            {
                var obj = visionMsg.Objects(i).Value;
                string[] parts = obj.Label.Split('|');
                
                float scaleW = parts.Length > 2 ? float.Parse(parts[2]) : 1.0f;
                float scaleH = parts.Length > 3 ? float.Parse(parts[3]) : 1.0f;

                Vector3 finalWorldPosition = camTransform.TransformPoint(new Vector3(obj.Position3d.Value.X, obj.Position3d.Value.Y, obj.Position3d.Value.Z));

                furnitureManager.AddFurniture(new FurnitureManager.FurnitureData {
                    id = (int)obj.Id, 
                    label = parts[0], // 라벨은 디버깅용으로 놔둠 (tv, cat 등)
                    
                    // 🔥 핵심 수정: C++이 보내는 쓰레기 이름(parts[1])은 완전히 개나 줘버립니다!!
                    // 무조건 TRELLIS가 방금 만들어낸 "furniture_{고유ID}.ply" 를 강제로 찾게 만듭니다.
                    plyPath = $"furniture_{obj.Id}.ply", 
                    
                    position = finalWorldPosition,
                    scale = new Vector3(scaleW, scaleH, scaleW)
                });
            }
        }
        catch (Exception ex) { Debug.LogError($"[ZMQ] 데이터 분석 에러: {ex}"); }
    }
}