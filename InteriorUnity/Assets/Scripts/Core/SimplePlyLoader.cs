// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Rendering/
// 파일 명: SimplePlyLoader.cs

using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System;

public class SimplePlyLoader : MonoBehaviour
{
    public Material pointMaterial; 

    public void LoadPly(string filePath)
    {
        if (!File.Exists(filePath)) return;

        using (FileStream fs = File.Open(filePath, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
        {
            int vertexCount = 0;
            int vertexSize = 0;
            Dictionary<string, int> propOffsets = new Dictionary<string, int>();

            string line;
            while ((line = ReadLine(reader)) != "end_header")
            {
                if (line.StartsWith("element vertex"))
                    vertexCount = int.Parse(line.Split(' ')[2]);
                else if (line.StartsWith("property"))
                {
                    string[] parts = line.Split(' ');
                    if (parts.Length >= 3) {
                        propOffsets[parts[2]] = vertexSize;
                        vertexSize += 4;
                    }
                }
            }

            byte[] vData = reader.ReadBytes(vertexCount * vertexSize);

            Vector3[] vertices = new Vector3[vertexCount];
            Color[] colors = new Color[vertexCount];
            int[] indices = new int[vertexCount];

            for (int i = 0; i < vertexCount; i++)
            {
                int baseIdx = i * vertexSize;

                // 1. PLY 내부의 로컬 원점 좌표 읽기
                Vector3 rawPos = new Vector3(
                    GetFloat(vData, propOffsets, baseIdx, "x"),
                    -GetFloat(vData, propOffsets, baseIdx, "y"),
                    GetFloat(vData, propOffsets, baseIdx, "z")
                );

                // 🔥 2. 위치 잡는 완벽한 로직 유지: 버퍼에 넣기 전에 월드 좌표로 구워버림!
                vertices[i] = transform.TransformPoint(rawPos);

                float r = GetFloat(vData, propOffsets, baseIdx, "f_dc_0");
                float g = GetFloat(vData, propOffsets, baseIdx, "f_dc_1");
                float b = GetFloat(vData, propOffsets, baseIdx, "f_dc_2");
                
                colors[i] = new Color(r, g, b, 1f);
                
                // 순수 점(Point) 렌더링을 위한 인덱스
                indices[i] = i; 
            }

            Mesh mesh = new Mesh();
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32; 
            mesh.SetVertices(vertices);
            mesh.SetColors(colors);
            
            // 🔥 3. 점선면(Quad) 절대 안 씀! 오직 촘촘한 점(Points) 위상으로만 출력!
            mesh.SetIndices(indices, MeshTopology.Points, 0);

            // 🔥 4. 투명화/컬링 방지: 점 위치가 바뀌었으니 바운딩 박스 재계산!
            mesh.RecalculateBounds(); 

            MeshFilter mf = GetComponent<MeshFilter>();
            if (mf == null) mf = gameObject.AddComponent<MeshFilter>();
            mf.mesh = mesh;

            MeshRenderer mr = GetComponent<MeshRenderer>();
            if (mr == null) mr = gameObject.AddComponent<MeshRenderer>();

            // 🔥 5. 투명화 원인 완벽 제거: 점을 그릴 수 있는 유니티 기본 정점 셰이더를 강제 주입하여 
            // F를 누르지 않아도 무조건 선명하게 눈에 보이도록 만듭니다!
            mr.material = new Material(Shader.Find("Sprites/Default"));

            // 이미 월드 좌표로 점들을 옮겼으므로 트랜스폼은 원점 초기화
            transform.position = Vector3.zero;
            transform.rotation = Quaternion.identity;
            transform.localScale = Vector3.one;

            Debug.Log($"[SimplePlyLoader] 🎉 {vertexCount}개의 순수 점(Point) 렌더링 및 가시화 완료!");
        }
    }

    private float GetFloat(byte[] data, Dictionary<string, int> offsets, int baseIdx, string key) {
        if (offsets.TryGetValue(key, out int offset))
            return BitConverter.ToSingle(data, baseIdx + offset);
        return 0f;
    }

    private string ReadLine(BinaryReader reader)
    {
        StringBuilder sb = new StringBuilder();
        char c;
        try {
            while ((c = reader.ReadChar()) != '\n') if (c != '\r') sb.Append(c);
        } catch { }
        return sb.ToString();
    }
}