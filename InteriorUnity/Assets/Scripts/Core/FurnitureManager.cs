// Assets/Scripts/Core/FurnitureManager.cs
using UnityEngine;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;

public class FurnitureManager : MonoBehaviour
{
    [Header("Rendering Settings")]
    public Material pointMaterial; 
    
    // 🔥 유니티 Inspector에서 실시간으로 조절할 수 있는 변수들!
    [Header("3DGS Transform Settings")]
    public Vector3 plyScale = new Vector3(0.01f, 0.01f, 0.01f); // 원본이 너무 크면 0.01 등으로 줄이세요.
    public Vector3 plyRotation = new Vector3(180f, 0f, 0f);     // 뒤집힌 것을 바로잡기 위한 X축 180도 회전

    public ConcurrentQueue<FurnitureData> _furnitureQueue = new ConcurrentQueue<FurnitureData>();
    private Dictionary<int, GameObject> _spawnedFurniture = new Dictionary<int, GameObject>();
    private string _sharedAssetsFolder;

    public struct FurnitureData {
        public int id;
        public string label;
        public string plyPath; 
        public Vector3 position;
    }

    void Start()
    {
        _sharedAssetsFolder = @"C:\Users\hk100\Desktop\InteriorPlatform\shared\assets_3dgs";
    }

    void Update()
    {
        while (_furnitureQueue.TryDequeue(out FurnitureData data))
        {
            RenderFurniture(data);
        }
    }

    private void RenderFurniture(FurnitureData data)
    {
        string fullFilePath = Path.Combine(_sharedAssetsFolder, data.plyPath);
        
        if (File.Exists(fullFilePath))
        {
            SpawnPointCloud(data, fullFilePath);
        }
    }

    private void SpawnPointCloud(FurnitureData data, string filePath)
    {
        if (!_spawnedFurniture.ContainsKey(data.id))
        {
            GameObject plyObject = new GameObject($"{data.label}_{data.id}_3DGS");
            
            // 1. 위치 설정
            plyObject.transform.position = data.position;
            
            // 2. 크기 및 회전 설정 적용 (여기서 거대한 크기와 뒤집힘을 해결합니다!)
            plyObject.transform.localScale = plyScale;
            plyObject.transform.rotation = Quaternion.Euler(plyRotation);
            
            Mesh plyMesh = SimplePlyLoader.LoadPly(filePath);
            
            if (plyMesh != null)
            {
                MeshFilter filter = plyObject.AddComponent<MeshFilter>();
                MeshRenderer renderer = plyObject.AddComponent<MeshRenderer>();
                
                filter.mesh = plyMesh;
                renderer.material = pointMaterial != null ? pointMaterial : new Material(Shader.Find("Particles/Standard Unlit"));
                
                _spawnedFurniture.Add(data.id, plyObject);
                Debug.Log($"🎉 [{data.label}] 3D 점 구름 소환 완료!");
            }
        }
        else
        {
            // 이미 생성된 객체라면 위치만 업데이트
            _spawnedFurniture[data.id].transform.position = data.position;
        }
    }
}