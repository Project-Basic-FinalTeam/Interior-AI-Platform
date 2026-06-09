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

            Vector3[] rawVertices = new Vector3[vertexCount];
            Vector3 center = Vector3.zero;

            for (int i = 0; i < vertexCount; i++)
            {
                int baseIdx = i * vertexSize;
                rawVertices[i] = new Vector3(
                    GetFloat(vData, propOffsets, baseIdx, "x"),
                    -GetFloat(vData, propOffsets, baseIdx, "y"),
                    GetFloat(vData, propOffsets, baseIdx, "z")
                );
                center += rawVertices[i];
            }
            center /= vertexCount;

            Vector3[] vertices = new Vector3[vertexCount];
            float maxBound = 0f;

            // 원점으로 옮기면서 모델의 '가장 긴 축'의 길이를 구합니다.
            for (int i = 0; i < vertexCount; i++)
            {
                vertices[i] = rawVertices[i] - center;
                maxBound = Mathf.Max(maxBound, Mathf.Abs(vertices[i].x), Mathf.Abs(vertices[i].y), Mathf.Abs(vertices[i].z));
            }

            // 🔥 초소형 미니미 탈출: AI가 만든 점들을 1m(1.0) 크기로 꽉 차게 팽창시킵니다.
            // 모델의 원래 비율은 100% 보존되면서, 크기만 1단위로 커집니다.
            if (maxBound > 0)
            {
                for (int i = 0; i < vertexCount; i++)
                {
                    vertices[i] /= (maxBound * 2f);
                }
            }

            Color[] colors = new Color[vertexCount];
            int[] indices = new int[vertexCount];

            for (int i = 0; i < vertexCount; i++)
            {
                int baseIdx = i * vertexSize;
                float r = GetFloat(vData, propOffsets, baseIdx, "f_dc_0");
                float g = GetFloat(vData, propOffsets, baseIdx, "f_dc_1");
                float b = GetFloat(vData, propOffsets, baseIdx, "f_dc_2");
                
                colors[i] = new Color(r, g, b, 1f);
                indices[i] = i; 
            }

            Mesh mesh = new Mesh();
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32; 
            mesh.SetVertices(vertices);
            mesh.SetColors(colors);
            
            mesh.SetIndices(indices, MeshTopology.Points, 0);
            mesh.RecalculateBounds(); 

            BoxCollider bc = gameObject.AddComponent<BoxCollider>();
            bc.center = mesh.bounds.center;
            bc.size = mesh.bounds.size;

            gameObject.layer = LayerMask.NameToLayer("Furniture");

            MeshFilter mf = GetComponent<MeshFilter>();
            if (mf == null) mf = gameObject.AddComponent<MeshFilter>();
            mf.mesh = mesh;

            MeshRenderer mr = GetComponent<MeshRenderer>();
            if (mr == null) mr = gameObject.AddComponent<MeshRenderer>();
            mr.material = new Material(Shader.Find("Sprites/Default"));

            transform.localPosition = Vector3.zero;
            transform.localRotation = Quaternion.identity;

            Debug.Log($"[SimplePlyLoader] 🎉 {vertexCount}개의 점 렌더링 및 1m 기본 뼈대 정규화 완료!");
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