// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Core/
// 파일 명: FurnitureManager.cs

using UnityEngine;
using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine.Networking;
using System.Collections;
using System.IO;

public class FurnitureManager : MonoBehaviour
{
    [Header("Rendering Settings")]
    public Material pointMaterial; 
    
    [Header("Network Settings")]
    public string serverUrl = "http://100.118.177.19:8000/assets/"; 

    private ConcurrentQueue<FurnitureData> _furnitureQueue = new ConcurrentQueue<FurnitureData>();
    private Dictionary<int, GameObject> _spawnedFurniture = new Dictionary<int, GameObject>();

    public struct FurnitureData {
        public int id;
        public string label;
        public string plyPath; 
        public Vector3 position;
        public Vector3 scale;
    }

    public void AddFurniture(FurnitureData data) {
        _furnitureQueue.Enqueue(data);
    }

    void Update() {
        while (_furnitureQueue.TryDequeue(out FurnitureData data)) {
            RenderFurniture(data);
        }
    }

    private void RenderFurniture(FurnitureData data) {
        if (!_spawnedFurniture.ContainsKey(data.id)) {
            StartCoroutine(DownloadAndSpawnPly(data));
        } else {
            _spawnedFurniture[data.id].transform.position = data.position;
            _spawnedFurniture[data.id].transform.localScale = data.scale;
        }
    }

    private IEnumerator DownloadAndSpawnPly(FurnitureData data) {
        
        string fileName = data.plyPath; 
        
        if (string.IsNullOrEmpty(fileName)) {
            fileName = $"asset_unknown_{data.id}.ply";
        }
        
        string downloadUrl = serverUrl + fileName + "?t=" + System.DateTime.Now.Ticks;
        
        using (UnityWebRequest www = UnityWebRequest.Get(downloadUrl)) {
            www.timeout = 180;
            yield return www.SendWebRequest();
            if (www.result == UnityWebRequest.Result.Success) {
                byte[] plyData = www.downloadHandler.data;
                
                string tempFilePath = Path.Combine(Application.persistentDataPath, $"temp_{data.id}.ply");
                File.WriteAllBytes(tempFilePath, plyData);
                
                GameObject plyObject = new GameObject($"{data.label}_{data.id}_PointMesh");

                SimplePlyLoader loader = plyObject.AddComponent<SimplePlyLoader>();
                loader.pointMaterial = pointMaterial;
                loader.LoadPly(tempFilePath);

                FurnitureClicker clicker = plyObject.AddComponent<FurnitureClicker>();
                clicker.furnitureId = data.id;
                
                // 백엔드에서 전달받은 진짜 월드 좌표 그대로 배치
                plyObject.transform.position = data.position; 

                // 🔥 핵심 해결 부분: 무조건 (1,1,1)을 넣던 것을 파이썬에서 계산한 원본 비율로 변경!!
                plyObject.transform.localScale = data.scale; 
                
                _spawnedFurniture.Add(data.id, plyObject);
                
                if (File.Exists(tempFilePath)) File.Delete(tempFilePath);
            }
            else {
                Debug.LogError($"[3DGS 다운로드 실패] 가구 ID {data.id} | URL: {downloadUrl} | 사유: {www.error}");
            }
        }
    }

    public void SwapFurniture(int targetId, string newPlyFileName, string newLabel) {
        if (_spawnedFurniture.TryGetValue(targetId, out GameObject oldObj)) {
            Vector3 originalPosition = oldObj.transform.position;
            Vector3 originalScale = oldObj.transform.localScale; // 🔥 스왑 시에도 기존 스케일 기억

            Destroy(oldObj);
            _spawnedFurniture.Remove(targetId);

            Debug.Log($"[FurnitureManager] 🔄 가구 스왑 진행 중... 원래 위치: {originalPosition}");

            AddFurniture(new FurnitureData {
                id = targetId,
                label = newLabel,
                plyPath = newPlyFileName, 
                position = originalPosition, 
                scale = originalScale // 🔥 새 가구에도 기존의 현실적인 비율을 그대로 적용!
            });
        } else {
            Debug.LogWarning($"[FurnitureManager] 스왑 실패: ID {targetId} 가구를 찾을 수 없습니다.");
        }
    }
}