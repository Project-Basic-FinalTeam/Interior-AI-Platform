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
                    
                    if (reqSocket.TryReceiveFrameString(TimeSpan.FromSeconds(30), out string initialResponse) && initialResponse.StartsWith("COUNT:"))
                    {
                        int objectCount = int.Parse(initialResponse.Split(':')[1]);
                        if (objectCount > 0)
                        {
                            reqSocket.SendFrame(System.Text.Encoding.UTF8.GetBytes("GIVE_ME_FINAL_DATA"));
                            if (reqSocket.TryReceiveFrameBytes(TimeSpan.FromSeconds(Math.Max(30, objectCount * 10)), out byte[] rawData))
                            {
                                _mainThreadActionQueue.Enqueue(rawData);
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
                    id = i,
                    label = parts[0],
                    plyPath = parts.Length > 1 ? parts[1] : "",
                    position = finalWorldPosition,
                    scale = new Vector3(scaleW, scaleH, scaleW)
                });
            }
        }
        catch (Exception ex) { Debug.LogError($"[ZMQ] 데이터 분석 에러: {ex}"); }
    }
}