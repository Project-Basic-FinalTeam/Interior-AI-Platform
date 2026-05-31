// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Network/
// 파일 명: ZmqReceiver.cs

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
    public RoomBuilder roomBuilder;

    private bool _isScanning = false;
    private ConcurrentQueue<byte[]> _mainThreadActionQueue = new ConcurrentQueue<byte[]>();

    void Awake()
{
    AsyncIO.ForceDotNet.Force();
    serverUrl = "tcp://100.118.177.19:5555".Trim();
    Debug.Log($"[ZMQ] 통신 모듈 초기화 완료: {serverUrl}");

    if (roomBuilder == null)
    {
        roomBuilder = FindAnyObjectByType<RoomBuilder>();

        if (roomBuilder == null)
        {
            GameObject rbObj = new GameObject("RoomBuilder");
            roomBuilder = rbObj.AddComponent<RoomBuilder>();
            Debug.Log("[ZMQ] RoomBuilder 오브젝트 자동 생성 완료");
        }
        else
        {
            Debug.Log("[ZMQ] RoomBuilder 자동 연결 성공");
        }
    }
}

    void OnDestroy()
    {
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

                    if (reqSocket.TryReceiveFrameString(TimeSpan.FromSeconds(30), out string initialResponse) && initialResponse.StartsWith("COUNT:"))
                    {
                        string[] parts = initialResponse.Split('|');
                        int objectCount = int.Parse(parts[0].Replace("COUNT:", ""));
                        string estTime = parts.Length > 1 ? parts[1].Replace("EST_SEC:", "") : "알 수 없음";

                        Debug.Log($"[3DGS] ✅ 스캔 완료! 총 {objectCount}개의 가구가 감지되었습니다. (렌더링 예상 대기 시간: 약 {estTime}초)");

                        if (objectCount > 0)
                        {
                            reqSocket.SendFrame(System.Text.Encoding.UTF8.GetBytes("GIVE_ME_FINAL_DATA"));

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
            catch (Exception e)
            {
                Debug.LogError($"[ZMQ] 통신 에러: {e.Message}");
            }
            finally
            {
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
                float targetScale = Mathf.Max(scaleW, scaleH);

                Vector3 finalWorldPosition = camTransform.TransformPoint(
                    new Vector3(obj.Position3d.Value.X, obj.Position3d.Value.Y, obj.Position3d.Value.Z));

                furnitureManager.AddFurniture(new FurnitureManager.FurnitureData {
                    id       = (int)obj.Id,
                    label    = parts[0],
                    plyPath  = $"furniture_{obj.Id}.ply",
                    position = finalWorldPosition,
                    scale    = new Vector3(targetScale, targetScale, targetScale)
                });
            }

            // ✅ 루프 완료 후 딱 한 번만 호출
            if (roomBuilder != null)
            {
                Debug.Log("[ZMQ] 가구 처리 완료 → RoomBuilder 트리거");
                roomBuilder.TriggerBuild();
            }
            else
            {
                Debug.LogError("[ZMQ] RoomBuilder를 찾을 수 없어 방 생성 불가");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[ZMQ] 데이터 분석 에러: {ex}");
        }
    }
}