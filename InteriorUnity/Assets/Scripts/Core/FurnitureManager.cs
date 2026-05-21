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
        string fileName = "furniture_" + data.id + ".ply";
        string downloadUrl = serverUrl + fileName + "?t=" + System.DateTime.Now.Ticks;
        
        using (UnityWebRequest www = UnityWebRequest.Get(downloadUrl)) {
            yield return www.SendWebRequest();
            if (www.result == UnityWebRequest.Result.Success) {
                byte[] plyData = www.downloadHandler.data;
                
                string tempFilePath = Path.Combine(Application.persistentDataPath, $"temp_{data.id}.ply");
                File.WriteAllBytes(tempFilePath, plyData);
                
                GameObject plyObject = new GameObject($"{data.label}_{data.id}_PointMesh");
                plyObject.transform.position = data.position;
                plyObject.transform.localScale = data.scale;
                
                SimplePlyLoader loader = plyObject.AddComponent<SimplePlyLoader>();
                loader.pointMaterial = pointMaterial;
                loader.LoadPly(tempFilePath);
                
                _spawnedFurniture.Add(data.id, plyObject);
                
                if (File.Exists(tempFilePath)) File.Delete(tempFilePath);
            }
        }
    }
}