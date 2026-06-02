// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Interaction/
// 파일 명: FurnitureClicker.cs

using UnityEngine;

public class FurnitureClicker : MonoBehaviour
{
    public int furnitureId;
    private ZmqReceiver _zmqReceiver;

    void Start()
    {
        // 🔥 유니티 최신 버전 권장 문법으로 변경 완료 (경고 해결)
        _zmqReceiver = FindAnyObjectByType<ZmqReceiver>();
    }

    void OnMouseDown()
    {
        if (_zmqReceiver != null)
        {
            Debug.Log($"[Interaction] 가구(ID: {furnitureId}) 클릭됨! C++로 교체(RAG) 요청을 보냅니다.");
            
            // 임시 하드코딩된 사용자 텍스트 쿼리
            string mockUserQuery = "이 가구 말고 더 모던한 화이트 톤으로 추천해줘.";
            
            // C++로 요청 날리기!
            _zmqReceiver.RequestRAGRecommendationFromCPP(furnitureId, mockUserQuery);
        }
        else
        {
            Debug.LogWarning("[Interaction] ZmqReceiver를 찾을 수 없습니다.");
        }
    }
}