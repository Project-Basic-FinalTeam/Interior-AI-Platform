// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Interaction/
// 파일 명: FurnitureRagSelector.cs

using UnityEngine;
using System.Threading.Tasks;
using NetMQ;
using NetMQ.Sockets;
using System.Collections.Concurrent;
using System;
using UnityEngine.InputSystem; 

public class FurnitureRagSelector : MonoBehaviour
{
    [Header("Dependencies")]
    public FurnitureManager furnitureManager; 
    
    [Header("Network")]
    // 🔥 localhost 대신 명확한 IPv4 주소를 적어줍니다.
    public string cplusplusServerUrl = "tcp://127.0.0.1:5555"; 

    private ConcurrentQueue<Action> _mainThreadActions = new ConcurrentQueue<Action>();

    void Update()
    {
        while (_mainThreadActions.TryDequeue(out Action action)) {
            action.Invoke();
        }

        if (Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame)
        {
            Vector2 mousePosition = Mouse.current.position.ReadValue();
            Ray ray = Camera.main.ScreenPointToRay(mousePosition);

            if (Physics.Raycast(ray, out RaycastHit hit, 100f))
            {
                string objName = hit.collider.gameObject.name;
                string[] parts = objName.Split('_');
                
                if (parts.Length >= 2 && int.TryParse(parts[1], out int targetId))
                {
                    Debug.Log($"<color=green>🎯 [RAG] 객체 선택됨! ID: {targetId}. 추천 서버에 분석을 요청합니다...</color>");
                    RequestRagRecommendationAsync(targetId);
                }
            }
        }
    }

    private void RequestRagRecommendationAsync(int targetId)
    {
        Task.Run(() =>
        {
            try {
                using (RequestSocket client = new RequestSocket())
                {
                    client.Connect(cplusplusServerUrl);
                    client.SendFrame($"ASK_RAG|{targetId}");
                    
                    Debug.Log("[RAG Client] ⏳ C++ 서버로 요청 전송 완료, 응답 대기 중...");
                    
                    // =================================================================
                    // 🔥 [핵심 수정] 무한 대기 방지! 15초 안에 대답 안 오면 강제 종료
                    // =================================================================
                    if (client.TryReceiveFrameString(TimeSpan.FromSeconds(15), out string responseJson))
                    {
                        Debug.Log($"[RAG Client] 💡 GPT 응답 도착:\n{responseJson}");

                        RagResponseData parsedData = JsonUtility.FromJson<RagResponseData>(responseJson);
                        string finalPly = string.IsNullOrEmpty(parsedData.recommended_ply) ? "asset_unknown.ply" : parsedData.recommended_ply;
                        string finalLabel = string.IsNullOrEmpty(parsedData.answer) ? "AI 추천 완료" : parsedData.answer;

                        _mainThreadActions.Enqueue(() => {
                            furnitureManager.SwapFurniture(targetId, finalPly, finalLabel);
                            Debug.Log($"<color=cyan>✨ [마법의 스왑] AI 추천 완료: {finalPly} 로 교체!</color>");
                        });
                    }
                    else
                    {
                        // 15초가 지나도 대답이 없으면 에러를 뿜고 빠져나옵니다.
                        Debug.LogError("[RAG Client] 🚨 15초 타임아웃! C++ 서버와 통신이 끊겼습니다. IP 주소가 127.0.0.1 이 맞는지 확인하세요.");
                    }
                }
            }
            catch (Exception e) {
                Debug.LogError($"[RAG 통신 에러] {e.Message}");
            }
        });
    }
}

[System.Serializable] 
public class RagResponseData
{
    public string status;
    public string answer;
    public string model;
    public string recommended_ply; 
}