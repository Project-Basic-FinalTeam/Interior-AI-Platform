// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Core/
// 파일 명: RoomBuilder.cs

using UnityEngine;
using System.IO;
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

        var mat = floorMaterial != null
            ? floorMaterial
            : new Material(Shader.Find("Standard"));
        if (floorMaterial == null) mat.color = new Color(0.85f, 0.85f, 0.85f); 
        floor.GetComponent<Renderer>().material = mat;

        Debug.Log($"[RoomBuilder] 바닥 생성 완료 | pos={pos} normal={normal}");
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
        float size = Mathf.Max(maxDist * 2.5f, 8f) / 10f;
        return new Vector3(size, 1f, size);
    }

    private void BuildWalls(RoomData data)
    {
        if (data.floor_wall_boundary_edges == null || data.floor_wall_boundary_edges.Length == 0)
        {
            Debug.LogWarning("[RoomBuilder] 벽 엣지 데이터 없음");
            return;
        }

        const float wallHeight = 3.0f;

        // ✅ 엣지 좌표들의 실제 중심점 계산 (카메라 기준 월드 좌표)
        Vector3 boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        foreach (var edge in data.floor_wall_boundary_edges)
        {
            Vector3 s = ToUnity(edge.point_3d_start_m);
            Vector3 e = ToUnity(edge.point_3d_end_m);
            boundsMin = Vector3.Min(boundsMin, Vector3.Min(s, e));
            boundsMax = Vector3.Max(boundsMax, Vector3.Max(s, e));
        }

        Vector3 roomCenter = (boundsMin + boundsMax) * 0.5f;
        float roomW = boundsMax.x - boundsMin.x;
        float roomD = boundsMax.z - boundsMin.z;
        float floorY = boundsMin.y;

        // ✅ 엣지 실제 좌표 기반으로 4면 벽 생성
        var walls = new (Vector3 pos, Quaternion rot, float width)[]
        {
            // 앞벽 (+Z 끝)
            (new Vector3(roomCenter.x, floorY + wallHeight * 0.5f, boundsMax.z),
            Quaternion.LookRotation(Vector3.back,    Vector3.up), roomW),
            // 뒷벽 (-Z 끝)
            (new Vector3(roomCenter.x, floorY + wallHeight * 0.5f, boundsMin.z),
            Quaternion.LookRotation(Vector3.forward, Vector3.up), roomW),
            // 오른벽 (+X 끝)
            (new Vector3(boundsMax.x,  floorY + wallHeight * 0.5f, roomCenter.z),
            Quaternion.LookRotation(Vector3.left,    Vector3.up), roomD),
            // 왼벽 (-X 끝)
            (new Vector3(boundsMin.x,  floorY + wallHeight * 0.5f, roomCenter.z),
            Quaternion.LookRotation(Vector3.right,   Vector3.up), roomD),
        };

        string[] wallNames = { "AI_Wall_Front", "AI_Wall_Back", "AI_Wall_Right", "AI_Wall_Left" };

        for (int i = 0; i < walls.Length; i++)
        {
            GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Quad);
            wall.name = wallNames[i];
            wall.transform.position   = walls[i].pos;
            wall.transform.localScale = new Vector3(walls[i].width, wallHeight, 1f);
            wall.transform.rotation   = walls[i].rot;

            var mat = wallMaterial != null
                ? wallMaterial
                : new Material(Shader.Find("Standard"));
            if (wallMaterial == null) mat.color = Color.white;
            wall.GetComponent<Renderer>().material = mat;

            Debug.Log($"[RoomBuilder] {wallNames[i]} 생성 | pos={walls[i].pos} | 너비={walls[i].width:F2}m");
        }
    }
}