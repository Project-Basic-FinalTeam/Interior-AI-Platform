// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Core/
// 파일 명: RoomBuilder.cs

using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Networking;

[Serializable] public class Point3D { public float x; public float y; public float z; }
[Serializable] public class WallEdge {
    public Point3D point_3d_start_m;
    public Point3D point_3d_end_m;
}
[Serializable] public class FloorPlane {
    public float[] plane_normal_camera_coord;
    public float plane_d;
}
[Serializable] public class RoomData {
    public FloorPlane floor_plane_estimation;
    public WallEdge[] floor_wall_boundary_edges;
}

public class RoomBuilder : MonoBehaviour
{
    public Material floorMaterial;
    public Material wallMaterial;

    [Header("JSON 서버 경로")]
    public string httpUrl = "http://100.118.177.19:8000/assets/depth_result/result.json";

    // ✅ ZmqReceiver로부터 전달받은 가구 월드 좌표 목록
    private List<Vector3> _furniturePositions = new List<Vector3>();

    void Start() { }

    // ✅ ZmqReceiver가 가구 위치 목록을 주입하는 메서드
    public void SetFurniturePositions(List<Vector3> positions)
    {
        _furniturePositions = positions;
        Debug.Log($"[RoomBuilder] 가구 위치 {positions.Count}개 수신 완료");
    }

    public void TriggerBuild()
    {
        StartCoroutine(LoadJsonViaHttp());
    }

    private IEnumerator LoadJsonViaHttp()
    {
        string url = httpUrl + "?t=" + DateTime.Now.Ticks;
        Debug.Log($"[RoomBuilder] JSON 요청: {url}");

        using (UnityWebRequest www = UnityWebRequest.Get(url))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("[RoomBuilder] ✅ JSON 수신 성공");
                ParseAndBuild(www.downloadHandler.text);
            }
            else
            {
                Debug.LogError($"[RoomBuilder] ❌ JSON 실패: {www.error} | URL: {url}");
            }
        }
    }

    private void ParseAndBuild(string json)
    {
        RoomData data = JsonUtility.FromJson<RoomData>(json);

        if (data?.floor_plane_estimation == null)
        {
            Debug.LogError("[RoomBuilder] 파싱 실패. JSON 구조 확인:\n" + json);
            return;
        }

        float[] n = data.floor_plane_estimation.plane_normal_camera_coord;
        float   d = data.floor_plane_estimation.plane_d;
        Debug.Log($"[RoomBuilder] 파싱 성공 | normal=({n[0]:F2},{n[1]:F2},{n[2]:F2}) d={d:F2} | 엣지={data.floor_wall_boundary_edges?.Length ?? 0}개");

        BuildFloor(data);
        BuildWalls(data);
    }

    private Vector3 ToUnity(float x, float y, float z) => new Vector3(x, -y, z);
    private Vector3 ToUnity(Point3D p) => ToUnity(p.x, p.y, p.z);

    private Material CreateMaterial(Color color)
    {
        string[] shaderCandidates = {
            "Universal Render Pipeline/Lit",
            "Universal Render Pipeline/Unlit",
            "Unlit/Color",
            "Standard",
            "Sprites/Default"
        };

        Shader shader = null;
        foreach (var name in shaderCandidates)
        {
            shader = Shader.Find(name);
            if (shader != null)
            {
                Debug.Log($"[RoomBuilder] 셰이더 선택: {name}");
                break;
            }
        }

        if (shader == null)
        {
            Debug.LogError("[RoomBuilder] 사용 가능한 셰이더 없음");
            return new Material(Shader.Find("Hidden/InternalErrorShader"));
        }

        Material mat = new Material(shader);
        mat.color = color;
        return mat;
    }

    private void BuildFloor(RoomData data)
    {
        float[] n  = data.floor_plane_estimation.plane_normal_camera_coord;
        float   d  = data.floor_plane_estimation.plane_d;
        Vector3 normal = ToUnity(n[0], n[1], n[2]).normalized;
        Vector3 pos    = -d * normal;

        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "AI_Generated_Floor";
        floor.transform.position   = pos;
        floor.transform.up         = normal;
        floor.transform.localScale = CalcFloorScale(data);

        floor.GetComponent<Renderer>().material = floorMaterial != null
            ? floorMaterial
            : CreateMaterial(new Color(0.85f, 0.85f, 0.85f));

        Debug.Log($"[RoomBuilder] 바닥 생성 완료 | pos={pos} | normal={normal}");
    }

    private Vector3 CalcFloorScale(RoomData data)
    {
        if (data.floor_wall_boundary_edges == null || data.floor_wall_boundary_edges.Length == 0)
            return new Vector3(1f, 1f, 1f);

        float maxDist = 0f;
        foreach (var edge in data.floor_wall_boundary_edges)
        {
            maxDist = Mathf.Max(maxDist,
                ToUnity(edge.point_3d_start_m).magnitude,
                ToUnity(edge.point_3d_end_m).magnitude);
        }
        float size = Mathf.Max(maxDist * 3.5f, 10f) / 10f;
        return new Vector3(size, 1f, size);
    }

    private void BuildWalls(RoomData data)
    {
        if (data.floor_wall_boundary_edges == null || data.floor_wall_boundary_edges.Length == 0)
        {
            Debug.LogWarning("[RoomBuilder] 벽 엣지 데이터 없음");
            return;
        }

        const float wallHeight = 8.0f;
        const float wallThick  = 0.1f;

        Vector3 floorScale = CalcFloorScale(data);
        float roomW = floorScale.x * 10f;
        float roomD = floorScale.z * 10f;

        float[] n    = data.floor_plane_estimation.plane_normal_camera_coord;
        float   d    = data.floor_plane_estimation.plane_d;
        Vector3 normal   = ToUnity(n[0], n[1], n[2]).normalized;
        Vector3 floorPos = -d * normal;

        float floorY  = floorPos.y;
        float centerX = floorPos.x;
        float centerZ = floorPos.z;
        float halfW   = roomW * 0.5f;
        float halfD   = roomD * 0.5f;

        // ✅ 가구 위치 기준으로 벽이 항상 가구 뒤에 오도록 경계 보정
        if (_furniturePositions != null && _furniturePositions.Count > 0)
        {
            float maxFurnitureZ = float.MinValue;
            float maxFurnitureX = float.MinValue;
            float minFurnitureX = float.MaxValue;

            foreach (var fp in _furniturePositions)
            {
                maxFurnitureZ = Mathf.Max(maxFurnitureZ, fp.z);
                maxFurnitureX = Mathf.Max(maxFurnitureX, fp.x);
                minFurnitureX = Mathf.Min(minFurnitureX, fp.x);
            }

            // 가구 최대 Z보다 뒷벽이 앞에 오면 → 가구 뒤로 밀어냄
            float backWallZ = centerZ + halfD;
            if (backWallZ < maxFurnitureZ + 1.0f)
            {
                float diff = (maxFurnitureZ + 1.0f) - backWallZ;
                halfD += diff;
                Debug.Log($"[RoomBuilder] 뒷벽 보정: +{diff:F2}m (가구 최대Z={maxFurnitureZ:F2})");
            }

            // 가구 최대/최소 X보다 좌우벽이 안쪽에 오면 → 바깥으로 밀어냄
            float rightWallX = centerX + halfW;
            float leftWallX  = centerX - halfW;
            if (rightWallX < maxFurnitureX + 0.5f)
            {
                float diff = (maxFurnitureX + 0.5f) - rightWallX;
                halfW += diff;
                Debug.Log($"[RoomBuilder] 오른벽 보정: +{diff:F2}m");
            }
            if (leftWallX > minFurnitureX - 0.5f)
            {
                float diff = leftWallX - (minFurnitureX - 0.5f);
                halfW += diff;
                Debug.Log($"[RoomBuilder] 왼벽 보정: +{diff:F2}m");
            }

            roomW = halfW * 2f;
            roomD = halfD * 2f;
        }

        Color woodColor = new Color(0.72f, 0.53f, 0.35f);

        CreateWallCube("AI_Wall_Back",
            new Vector3(centerX,         floorY + wallHeight * 0.5f, centerZ + halfD),
            new Vector3(roomW, wallHeight, wallThick),
            woodColor);

        CreateWallCube("AI_Wall_Right",
            new Vector3(centerX + halfW,  floorY + wallHeight * 0.5f, centerZ),
            new Vector3(wallThick, wallHeight, roomD),
            woodColor);

        CreateWallCube("AI_Wall_Left",
            new Vector3(centerX - halfW,  floorY + wallHeight * 0.5f, centerZ),
            new Vector3(wallThick, wallHeight, roomD),
            woodColor);
    }

    private void CreateWallCube(string wallName, Vector3 pos, Vector3 scale, Color color)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = wallName;
        wall.transform.position   = pos;
        wall.transform.localScale = scale;
        wall.transform.rotation   = Quaternion.identity;

        wall.GetComponent<Renderer>().material = wallMaterial != null
            ? wallMaterial
            : CreateMaterial(color);

        Debug.Log($"[RoomBuilder] {wallName} 생성 | pos={pos} | scale={scale}");
    }
}