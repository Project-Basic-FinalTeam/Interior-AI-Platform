// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Core/
// 파일 명: RoomBuilder.cs

using UnityEngine;
using System;
using System.Collections;
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

    void Start() { }

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

    // ✅ 분홍색 방지: 프로젝트 렌더 파이프라인에 맞는 셰이더 자동 탐색
    private Material CreateMaterial(Color color)
    {
        // URP → Built-in 순서로 사용 가능한 셰이더 탐색
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
            Debug.LogError("[RoomBuilder] 사용 가능한 셰이더 없음 - Material 생성 실패");
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

        // ✅ Inspector에 Material이 연결되면 그걸 쓰고, 없으면 코드로 생성
        floor.GetComponent<Renderer>().material = floorMaterial != null
            ? floorMaterial
            : CreateMaterial(new Color(0.85f, 0.85f, 0.85f)); // 연한 회색

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

        const float wallHeight = 30.0f;
        const float wallThick  = 0.1f; // Cube 두께

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

        Color woodColor = new Color(0.72f, 0.53f, 0.35f);

        // ✅ Cube로 벽 생성 (양면 보임, 두께 있음)
        // 뒷벽: Z+ 끝, X방향으로 늘림
        CreateWallCube("AI_Wall_Back",
            new Vector3(centerX,        floorY + wallHeight * 0.5f, centerZ + halfD),
            new Vector3(roomW, wallHeight, wallThick),
            woodColor);

        // 앞벽: Z- 끝
        CreateWallCube("AI_Wall_Front",
            new Vector3(centerX,        floorY + wallHeight * 0.5f, centerZ - halfD),
            new Vector3(roomW, wallHeight, wallThick),
            woodColor);

        // 오른벽: X+ 끝, Z방향으로 늘림
        CreateWallCube("AI_Wall_Right",
            new Vector3(centerX + halfW, floorY + wallHeight * 0.5f, centerZ),
            new Vector3(wallThick, wallHeight, roomD),
            woodColor);

        // 왼벽: X- 끝
        CreateWallCube("AI_Wall_Left",
            new Vector3(centerX - halfW, floorY + wallHeight * 0.5f, centerZ),
            new Vector3(wallThick, wallHeight, roomD),
            woodColor);
    }

    private void CreateWallCube(string wallName, Vector3 pos, Vector3 scale, Color color)
    {
        // ✅ Cube는 회전 불필요 - scale로 방향 결정
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