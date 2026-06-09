// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Interaction/
// 파일 명: HandFurnitureGrabber.cs

using UnityEngine;

public class HandFurnitureGrabber : MonoBehaviour
{
    [Header("MediaPipe Hand Landmarks")]
    [Tooltip("MediaPipe의 엄지손가락 끝(Thumb Tip - 번호 4) Transform을 연결하세요.")]
    public Transform thumbTip;
    
    [Tooltip("MediaPipe의 검지손가락 끝(Index Finger Tip - 번호 8) Transform을 연결하세요.")]
    public Transform indexTip;

    [Header("Grab Settings")]
    [Tooltip("엄지와 검지가 이 거리보다 가까워지면 '잡기(Pinch)'로 인식합니다.")]
    public float pinchThreshold = 0.05f; 
    
    [Tooltip("손가락 주변 몇 m 반경 안의 가구를 잡을 것인지 설정합니다.")]
    public float grabRadius = 0.15f; 
    
    [Tooltip("가구 객체만 잡기 위해 'Furniture' 레이어를 선택하세요.")]
    public LayerMask furnitureLayer;

    private GameObject _grabbedFurniture = null;
    private Vector3 _grabOffset;
    private bool _isPinching = false;

    void Update()
    {
        if (thumbTip == null || indexTip == null) return;

        // 1. 꼬집기(Pinch) 상태 판별: 엄지 끝과 검지 끝의 거리 계산
        float distance = Vector3.Distance(thumbTip.position, indexTip.position);
        _isPinching = distance < pinchThreshold;

        // 2. 꼬집는 순간 + 잡고 있는 가구가 없을 때 -> 가구 잡기 시도
        if (_isPinching && _grabbedFurniture == null)
        {
            TryGrabFurniture();
        }
        // 3. 꼬집고 있는 상태 + 잡고 있는 가구가 있을 때 -> 손을 따라 가구 이동
        else if (_isPinching && _grabbedFurniture != null)
        {
            MoveGrabbedFurniture();
        }
        // 4. 손가락을 폈을 때 -> 가구 놓기
        else if (!_isPinching && _grabbedFurniture != null)
        {
            ReleaseFurniture();
        }
    }

    private void TryGrabFurniture()
    {
        // Furniture 레이어만 딱 지정해서 검사
        Collider[] targetHits = Physics.OverlapSphere(indexTip.position, grabRadius, furnitureLayer);

        if (targetHits.Length > 0)
        {
            _grabbedFurniture = targetHits[0].gameObject;
            _grabOffset = _grabbedFurniture.transform.position - GetPinchCenter();
            Debug.Log($"<color=yellow>🖐️ 최종 잡기 성공! 가구 이름: {_grabbedFurniture.name}</color>");
        }
        // else { } <-- 아무것도 없을 때 로그를 찍지 않아서 스팸 방지!
    }                               

    private void MoveGrabbedFurniture()
    {
        // 꼬집은 중앙 지점 + 오프셋 위치로 가구 실시간 이동
        _grabbedFurniture.transform.position = Vector3.Lerp(
            _grabbedFurniture.transform.position, 
            GetPinchCenter() + _grabOffset, 
            Time.deltaTime * 15f // 부드러운 이동을 위해 Lerp 적용
        );
    }

    private void ReleaseFurniture()
    {
        Debug.Log($"[Interaction] ✋ 가구 놓음: {_grabbedFurniture.name}");
        _grabbedFurniture = null;
    }

    // 엄지와 검지의 정중앙 좌표를 구하는 헬퍼 함수
    private Vector3 GetPinchCenter()
    {
        return (thumbTip.position + indexTip.position) / 2f;
    }

    // Unity 에디터에서 잡기 반경을 시각적으로 확인하기 위한 기즈모
    private void OnDrawGizmos()
    {
        if (indexTip != null)
        {
            Gizmos.color = _isPinching ? Color.red : Color.green;
            Gizmos.DrawWireSphere(indexTip.position, grabRadius);
        }
    }
}