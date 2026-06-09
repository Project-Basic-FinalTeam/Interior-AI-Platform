// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Interaction/
// 파일 명: FurnitureClicker.cs

using UnityEngine;

public class FurnitureClicker : MonoBehaviour
{
    public int furnitureId;
    private ZmqReceiver _zmqReceiver;

    [Header("Mouse Interaction Settings")]
    [Tooltip("마우스 휠 회전 속도")]
    public float rotationSpeed = 1000f;
    
    [Tooltip("W/S 키를 이용한 앞뒤(깊이) 이동 속도")]
    public float depthSpeed = 5f; // 🔥 새로 추가된 깊이 조절 속도

    // 마우스 드래그 상태 관리 변수
    private bool _isDragging = false;
    private Vector3 _dragOffset;
    private float _zCoord;

    void Start()
    {
        _zmqReceiver = FindAnyObjectByType<ZmqReceiver>();
    }

    void Update()
    {
        // 1. 드래그(선택) 중일 때 마우스 휠 및 키보드 조작
        if (_isDragging)
        {
            // [회전] 마우스 휠로 좌/우 회전
            float scroll = Input.GetAxis("Mouse ScrollWheel");
            if (Mathf.Abs(scroll) > 0.01f)
            {
                transform.Rotate(Vector3.up, scroll * rotationSpeed * Time.deltaTime, Space.World);
            }

            // 🔥 [깊이 조절] 키보드 W(멀어짐) / S(가까워짐)
            bool depthChanged = false;
            if (Input.GetKey(KeyCode.W))
            {
                _zCoord += depthSpeed * Time.deltaTime; // 카메라에서 멀어짐
                depthChanged = true;
            }
            else if (Input.GetKey(KeyCode.S))
            {
                _zCoord -= depthSpeed * Time.deltaTime; // 카메라와 가까워짐
                _zCoord = Mathf.Max(0.5f, _zCoord);     // 가구가 카메라 화면 뒤로 넘어가는 것 방지
                depthChanged = true;
            }

            // 키보드를 눌러 깊이가 변했다면, 마우스를 움직이지 않아도 즉시 위치를 갱신합니다.
            if (depthChanged)
            {
                transform.position = GetMouseAsWorldPoint() + _dragOffset;
            }
        }

        // 2. 우클릭(1) 시 RAG 가구 교체 요청
        if (Input.GetMouseButtonDown(1))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                if (hit.collider.gameObject == this.gameObject)
                {
                    RequestRAG();
                }
            }
        }
    }

    // 마우스 좌클릭 누를 때 (드래그 시작)
    void OnMouseDown()
    {
        _zCoord = Camera.main.WorldToScreenPoint(gameObject.transform.position).z;
        _dragOffset = gameObject.transform.position - GetMouseAsWorldPoint();
        _isDragging = true;
    }

    // 마우스 좌클릭 누른 채로 움직일 때 (드래그 중)
    void OnMouseDrag()
    {
        transform.position = GetMouseAsWorldPoint() + _dragOffset;
    }

    // 마우스 좌클릭 뗄 때 (드래그 종료)
    void OnMouseUp()
    {
        _isDragging = false;
    }

    // 마우스의 2D 스크린 좌표를 3D 월드 좌표로 변환하는 헬퍼 함수
    private Vector3 GetMouseAsWorldPoint()
    {
        Vector3 mousePoint = Input.mousePosition;
        mousePoint.z = _zCoord; // 현재 계산된 깊이값(_zCoord)을 반영
        return Camera.main.ScreenToWorldPoint(mousePoint);
    }

    // RAG 추천 요청 (우클릭 시 실행)
    private void RequestRAG()
    {
        if (_zmqReceiver != null)
        {
            Debug.Log($"<color=orange>[Interaction] 가구(ID: {furnitureId}) 우클릭됨! C++로 RAG 교체 요청을 보냅니다.</color>");
            string mockUserQuery = "이 가구 말고 더 모던한 화이트 톤으로 추천해줘.";
            _zmqReceiver.RequestRAGRecommendationFromCPP(furnitureId, mockUserQuery);
        }
        else
        {
            Debug.LogWarning("[Interaction] ZmqReceiver를 찾을 수 없습니다.");
        }
    }
}