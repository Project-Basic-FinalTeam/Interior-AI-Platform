using UnityEngine;
using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine.Networking;
using System.Collections;

public class FurnitureManager : MonoBehaviour
{
    [Header("Rendering Settings")]
    public Material pointMaterial; 
    
    [Header("3DGS Transform Settings")]
    // 기존의 고정 plyScale은 이제 파이썬 동적 스케일로 대체되므로 사용하지 않습니다.
    public Vector3 plyRotation = new Vector3(180f, 0f, 0f); 

    // 서버(FastAPI) 주소 설정
    [Header("Network Settings")]
    public string serverUrl = "http://100.118.177.19:8000/assets/"; 

    public ConcurrentQueue<FurnitureData> _furnitureQueue = new ConcurrentQueue<FurnitureData>();
    private Dictionary<int, GameObject> _spawnedFurniture = new Dictionary<int, GameObject>();

    public struct FurnitureData {
        public int id;
        public string label;
        // plyPath는 이제 쓰지 않거나 무시해도 됩니다. ID로 URL을 조합합니다.
        public string plyPath; 
        public Vector3 position;
        
        // 🔥 [추가 항목] 파이썬에서 ZMQ를 통해 전달받은 실제 물리적 크기(미터 단위)
        public Vector3 scale; 
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
        if (!_spawnedFurniture.ContainsKey(data.id))
        {
            // 서버에서 파일을 다운로드하는 코루틴 시작
            StartCoroutine(DownloadAndSpawnPly(data));
        }
        else
        {
            // 이미 생성된 객체라면 위치와 크기만 최신화
            _spawnedFurniture[data.id].transform.position = data.position;
            _spawnedFurniture[data.id].transform.localScale = data.scale;
        }
    }

    private IEnumerator DownloadAndSpawnPly(FurnitureData data)
    {
        // 1. 객체 ID를 바탕으로 다운로드 URL 조합 (예: http://100.118.177.19:8000/assets/furniture_0.ply)
        string fileName = "furniture_" + data.id + ".ply";
        // 🔥 [캐시 무력화] 주소 뒤에 "?t=시간"을 붙여서 유니티가 매번 무조건 새로 다운받게 강제합니다!
        string downloadUrl = serverUrl + fileName + "?t=" + System.DateTime.Now.Ticks;
        
        Debug.Log($"[FurnitureManager] 📥 서버에서 3DGS 파일 다운로드 시도: {downloadUrl}");

        using (UnityWebRequest www = UnityWebRequest.Get(downloadUrl))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.ConnectionError || www.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError($"[FurnitureManager] ❌ 다운로드 실패 ({fileName}): {www.error}");
                // 파일이 아직 다 안 만들어졌을 수도 있으니 나중에 재시도하는 로직을 넣어도 됩니다.
            }
            else
            {
                Debug.Log($"[FurnitureManager] ✅ 다운로드 성공! ({fileName}) -> 렌더링 시작");
                
                // 다운로드 받은 바이너리 데이터
                byte[] plyData = www.downloadHandler.data;

                // 다운로드 받은 데이터를 파싱하여 3D 객체 생성
                SpawnPointCloudFromMemory(data, plyData);
            }
        }
    }

    // 💡 기존의 파일 경로 방식이 아니라, 메모리(byte[])에서 바로 읽어오도록 수정
    private void SpawnPointCloudFromMemory(FurnitureData data, byte[] plyData)
    {
        GameObject plyObject = new GameObject($"{data.label}_{data.id}_3DGS");
        
        // 위치 지정
        plyObject.transform.position = data.position;
        
        // 🔥 [가장 핵심!] 정규화된 1x1 찌그러진 크기 대신, 파이썬이 보내준 진짜 현실 스케일 적용
        plyObject.transform.localScale = data.scale;
        
        // 회전 지정
        plyObject.transform.rotation = Quaternion.Euler(plyRotation);
        
        // 주의: SimplePlyLoader가 파일 경로(string)만 받는다면 수정이 필요합니다.
        // 임시로 다운로드 받은 데이터를 Application.persistentDataPath 에 저장하고 읽는 방식을 씁니다.
        string tempFilePath = System.IO.Path.Combine(Application.persistentDataPath, $"temp_{data.id}.ply");
        System.IO.File.WriteAllBytes(tempFilePath, plyData);

        Mesh plyMesh = SimplePlyLoader.LoadPly(tempFilePath);
        
        if (plyMesh != null)
        {
            MeshFilter filter = plyObject.AddComponent<MeshFilter>();
            MeshRenderer renderer = plyObject.AddComponent<MeshRenderer>();
            
            filter.mesh = plyMesh;
            renderer.material = pointMaterial != null ? pointMaterial : new Material(Shader.Find("Sprites/Default"));
            
            _spawnedFurniture.Add(data.id, plyObject);
            Debug.Log($"🎉 [{data.label}] 진짜 3DGS 점 구름 소환 완료! (크기: {data.scale.x:F2}m x {data.scale.y:F2}m)");
        }

        // 임시 파일 삭제 (선택)
        if (System.IO.File.Exists(tempFilePath)) {
            System.IO.File.Delete(tempFilePath);
        }
    }
}