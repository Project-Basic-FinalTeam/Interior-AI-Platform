// Assets/Scripts/Core/FurnitureManager.cs
using UnityEngine;
using System.Collections.Concurrent;
using System.Collections.Generic;

public class FurnitureManager : MonoBehaviour
{
    public GameObject plantPrefab; 
    // 큐가 정상적으로 초기화되었는지 확인
    public ConcurrentQueue<FurnitureData> _furnitureQueue = new ConcurrentQueue<FurnitureData>();
    private Dictionary<int, GameObject> _spawnedFurniture = new Dictionary<int, GameObject>();

    public struct FurnitureData {
        public int id;
        public string label;
        public Vector3 position;
    }

    void Start()
    {
        Debug.Log("[Manager] 가구 매니저 시작됨!");
    }

    void Update()
    {
        // 1. 매 프레임마다 큐의 상태를 체크 (너무 자주 뜨면 이 로그만 지우세요)
        if (_furnitureQueue.Count > 0)
        {
            Debug.Log($"[Manager] 현재 큐에 대기 중인 데이터 개수: {_furnitureQueue.Count}");
            
            while (_furnitureQueue.TryDequeue(out FurnitureData data))
            {
                Debug.Log($"[Manager] 메인 스레드에서 처리 시작: {data.label} (ID:{data.id})");
                RenderFurniture(data);
            }
        }
    }

    private void RenderFurniture(FurnitureData data)
    {
        if (plantPrefab == null) {
            Debug.LogError("[Manager] 에러: Plant Prefab이 할당되지 않았습니다! 인스펙터를 확인하세요.");
            return;
        }

        if (!_spawnedFurniture.ContainsKey(data.id))
        {
            Debug.Log($"🌱 [신규 생성] 위치: {data.position}");
            GameObject obj = Instantiate(plantPrefab, data.position, Quaternion.identity);
            obj.name = $"{data.label}_{data.id}";
            _spawnedFurniture.Add(data.id, obj);
        }
        else
        {
            Debug.Log($"🔄 [위치 업데이트] ID:{data.id} -> {data.position}");
            _spawnedFurniture[data.id].transform.position = data.position;
        }
    }
}