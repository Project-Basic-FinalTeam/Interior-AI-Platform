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
using System.Text;
using System.Collections.Generic;

public class ZmqReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    public string serverUrl = "tcp://100.118.177.19:5555";

    [Header("Manager Reference")]
    public FurnitureManager furnitureManager;
    public RoomBuilder roomBuilder;

    private bool _isScanning = false;

    // ✅ byte[] 큐와 Action 큐 통합 → Action 큐로 일원화
    private ConcurrentQueue<Action> _mainThreadActionQueue = new ConcurrentQueue<Action>();

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
        // ✅ 메인 스레드에서 Action 큐 처리
        while (_mainThreadActionQueue.TryDequeue(out Action action))
        {
            action?.Invoke();
        }
    }

    // ─────────────────────────────────────────
    // 기존: 방 스캔 및 가구 렌더링 요청
    // ─────────────────────────────────────────
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
                    reqSocket.SendFrame(Encoding.UTF8.GetBytes("SCAN_ROOM"));

                    if (reqSocket.TryReceiveFrameString(TimeSpan.FromSeconds(30), out string initialResponse)
                        && initialResponse.StartsWith("COUNT:"))
                    {
                        string[] parts = initialResponse.Split('|');
                        int objectCount = int.Parse(parts[0].Replace("COUNT:", ""));
                        string estTime  = parts.Length > 1 ? parts[1].Replace("EST_SEC:", "") : "알 수 없음";

                        Debug.Log($"[3DGS] ✅ 스캔 완료! 총 {objectCount}개의 가구가 감지되었습니다. (렌더링 예상 대기 시간: 약 {estTime}초)");

                        if (objectCount > 0)
                        {
                            reqSocket.SendFrame(Encoding.UTF8.GetBytes("GIVE_ME_FINAL_DATA"));

                            int waitTimeout = parts.Length > 1
                                ? int.Parse(estTime) + 30
                                : Math.Max(60, objectCount * 50);

                            if (reqSocket.TryReceiveFrameBytes(TimeSpan.FromSeconds(waitTimeout), out byte[] rawData))
                            {
                                // ✅ Action 큐에 바이트 처리 람다로 적재
                                _mainThreadActionQueue.Enqueue(() => ProcessScanResultOnMainThread(rawData));
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

    private void ProcessScanResultOnMainThread(byte[] rawData)
    {
        try
        {
            var bb       = new ByteBuffer(rawData);
            var visionMsg = VisionMessage.GetRootAsVisionMessage(bb);
            Transform camTransform = Camera.main.transform;

            List<Vector3> furnitureWorldPositions = new List<Vector3>();

            for (int i = 0; i < visionMsg.ObjectsLength; i++)
            {
                var obj    = visionMsg.Objects(i).Value;
                string[] parts = obj.Label.Split('|');

                float scaleW = parts.Length > 2 ? float.Parse(parts[2]) : 1.0f;
                float scaleH = parts.Length > 3 ? float.Parse(parts[3]) : 1.0f;
                float targetScale = Mathf.Max(scaleW, scaleH);

                Vector3 finalWorldPosition = camTransform.TransformPoint(
                    new Vector3(obj.Position3d.Value.X, obj.Position3d.Value.Y, obj.Position3d.Value.Z));

                furnitureWorldPositions.Add(finalWorldPosition);

                furnitureManager.AddFurniture(new FurnitureManager.FurnitureData {
                    id       = (int)obj.Id,
                    label    = parts[0],
                    plyPath  = $"furniture_{obj.Id}.ply",
                    position = finalWorldPosition,
                    scale    = new Vector3(targetScale, targetScale, targetScale)
                });
            }

            // ✅ 가구 루프 완료 후 딱 한 번만 RoomBuilder 트리거
            if (roomBuilder != null)
            {
                Debug.Log("[ZMQ] 가구 처리 완료 → RoomBuilder 트리거");
                roomBuilder.SetFurniturePositions(furnitureWorldPositions);
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

    // ─────────────────────────────────────────
    // 신규: RAG 기반 가구 교체 추천 요청
    // ─────────────────────────────────────────
    public void RequestRAGRecommendationFromCPP(int targetId, string userQuery)
    {
        Task.Run(() => {
            try
            {
                using (var reqSocket = new RequestSocket())
                {
                    reqSocket.Options.Linger = TimeSpan.FromSeconds(1);
                    reqSocket.Connect(serverUrl);

                    // 🔥 C++이 인식할 수 있도록 무조건 "ASK_RAG|ID|내용" 형태의 문자열로 보냅니다!
                    string requestStr = $"ASK_RAG|{targetId}|{userQuery}";
                    reqSocket.SendFrame(Encoding.UTF8.GetBytes(requestStr));

                    Debug.Log($"[ZMQ] C++ 코어로 RAG 추천 요청 전송: {requestStr}");

                    // 2. C++ 코어로부터 RAG 결과(JSON) 응답 대기
                    if (reqSocket.TryReceiveFrameString(TimeSpan.FromSeconds(180), out string responseJson))
                    {
                        Debug.Log($"[ZMQ] C++ 코어로부터 RAG 응답 수신: {responseJson}");
                        _mainThreadActionQueue.Enqueue(() => ProcessRAGResultOnMainThread(responseJson, targetId));
                    }
                    else
                    {
                        Debug.LogError("[ZMQ] 🚨 C++ RAG 응답 시간 초과! (C++ 서버가 죽었거나 RAG 엔진이 응답하지 않음)");
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[ZMQ] RAG 요청 통신 에러: {e.Message}");
            }
        });
    }

    private void ProcessRAGResultOnMainThread(string json, int targetId)
    {
        // [디버깅용] RAG에서 넘어온 원본 데이터를 출력
        Debug.LogWarning($"====================================\n[RAG 원본 데이터 수신 확인]\n길이: {json.Length}자\n내용:\n{json}\n====================================");

        try
        {
            RAGResponse response = JsonUtility.FromJson<RAGResponse>(json);

            // 상태가 success이거나, 파싱은 되었는데 status 필드가 누락되었을 수도 있으니 널 체크만 확실히 합니다.
            if (response != null)
            {
                // 🔥 [임시 방어 코드] JSON에 recommended_ply가 아예 없거나 비어있으면 강제로 micke.ply 할당!
                if (string.IsNullOrEmpty(response.recommended_ply))
                {
                    Debug.LogWarning("[RAG 강제 보정] JSON에 recommended_ply 값이 없습니다. 테스트를 위해 'micke.ply'로 강제 설정합니다.");
                    response.recommended_ply = "micke.ply";
                }

                // 만약 status가 비어있어도 진행하도록 조건 완화 (백엔드 JSON 포맷이 불안정하므로)
                Debug.Log($"[RAG 교체] AI의 추천: {response.answer}");
                furnitureManager.SwapFurniture(targetId, response.recommended_ply, "RAG_Recommended");
            }
            else
            {
                Debug.LogWarning("[RAG 교체 실패] JSON 파싱 결과가 Null입니다.");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[ZMQ] RAG JSON 파싱 에러: {ex.Message}\n에러를 일으킨 원본 문자열: {json}");
        }
    }

    // ─────────────────────────────────────────
    // JSON 파싱용 내부 클래스
    // ─────────────────────────────────────────
    [Serializable]
    public class RAGResponse
    {
        public string status;
        public string answer;
        public string recommended_ply;
        public string model;
    }
}