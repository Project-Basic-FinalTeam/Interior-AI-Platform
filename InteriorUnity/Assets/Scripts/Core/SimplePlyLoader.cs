using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

public class SimplePlyLoader : MonoBehaviour
{
    [StructLayout(LayoutKind.Sequential)]
    public struct SplatData
    {
        public Vector3 pos;
        public Vector3 color;
        public float opacity;
        public Vector3 scale;
        public Vector4 rot;
    }

    public Material pointMaterial; 
    private Material m_instanceMaterial; 
    private ComputeBuffer splatBuffer;
    private int splatCount = 0;
    
    private Mesh quadMesh;
    private MaterialPropertyBlock m_props; 

    public void LoadPly(string filePath)
    {
        if (!File.Exists(filePath)) return;

        List<SplatData> splatList = new List<SplatData>();
        
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

            for (int i = 0; i < vertexCount; i++)
            {
                int baseIdx = i * vertexSize;
                SplatData splat = new SplatData();

                splat.pos = new Vector3(
                    GetFloat(vData, propOffsets, baseIdx, "x"),
                    -GetFloat(vData, propOffsets, baseIdx, "y"),
                    GetFloat(vData, propOffsets, baseIdx, "z")
                );

                splat.color = new Vector3(
                    GetFloat(vData, propOffsets, baseIdx, "f_dc_0"),
                    GetFloat(vData, propOffsets, baseIdx, "f_dc_1"),
                    GetFloat(vData, propOffsets, baseIdx, "f_dc_2")
                );

                splat.opacity = GetFloat(vData, propOffsets, baseIdx, "opacity");

                splat.scale = new Vector3(
                    GetFloat(vData, propOffsets, baseIdx, "scale_0"),
                    GetFloat(vData, propOffsets, baseIdx, "scale_1"),
                    GetFloat(vData, propOffsets, baseIdx, "scale_2")
                );

                splat.rot = new Vector4(
                    GetFloat(vData, propOffsets, baseIdx, "rot_0"),
                    GetFloat(vData, propOffsets, baseIdx, "rot_1"),
                    GetFloat(vData, propOffsets, baseIdx, "rot_2"),
                    GetFloat(vData, propOffsets, baseIdx, "rot_3")
                );

                splatList.Add(splat);
            }
        }

        if (splatList.Count == 0) return;

        splatCount = splatList.Count;
        splatBuffer = new ComputeBuffer(splatCount, 56);
        splatBuffer.SetData(splatList.ToArray());
        
        if (pointMaterial != null) {
            m_instanceMaterial = new Material(pointMaterial);
            // [강제 바인딩] 매테리얼 자체에 버퍼 연결
            m_instanceMaterial.SetBuffer("_SplatBuffer", splatBuffer);
        }

        quadMesh = new Mesh();
        quadMesh.vertices = new Vector3[] {
            new Vector3(-1, -1, 0), new Vector3(1, -1, 0), new Vector3(1, 1, 0), new Vector3(-1, 1, 0)
        };
        quadMesh.triangles = new int[] { 0, 1, 2, 0, 2, 3 };
            
        Debug.Log($"[SimplePlyLoader] 🎉 {splatCount}개의 가구 데이터 파싱 완료!");
    }

    private float GetFloat(byte[] data, Dictionary<string, int> offsets, int baseIdx, string key) {
        if (offsets.TryGetValue(key, out int offset))
            return BitConverter.ToSingle(data, baseIdx + offset);
        return 0f;
    }

    void Update()
    {
        if (splatBuffer == null || m_instanceMaterial == null || quadMesh == null) return;

        // [이중 방어] 매 프레임 Material과 PropertyBlock 양쪽에 강제로 데이터를 밀어넣습니다.
        m_instanceMaterial.SetBuffer("_SplatBuffer", splatBuffer);
        m_instanceMaterial.SetMatrix("_CustomLocalToWorld", transform.localToWorldMatrix);

        if (m_props == null) m_props = new MaterialPropertyBlock();
        m_props.SetBuffer("_SplatBuffer", splatBuffer);
        m_props.SetMatrix("_CustomLocalToWorld", transform.localToWorldMatrix);

        Bounds bounds = new Bounds(transform.position, new Vector3(1000f, 1000f, 1000f)); 
        
        // [수정 완료] m_props를 전달하여 DX12 파이프라인이 매 프레임 올바른 SRV를 찾게 함
        Graphics.DrawMeshInstancedProcedural(
            quadMesh, 
            0, 
            m_instanceMaterial, 
            bounds, 
            splatCount, 
            m_props, // null -> m_props
            UnityEngine.Rendering.ShadowCastingMode.Off, 
            false
        );
    }

    void CleanUp()
    {
        if (splatBuffer != null) { splatBuffer.Release(); splatBuffer = null; }
        if (m_instanceMaterial != null) { Destroy(m_instanceMaterial); m_instanceMaterial = null; }
        if (quadMesh != null) { Destroy(quadMesh); quadMesh = null; }
    }

    void OnDisable() { CleanUp(); }
    void OnDestroy() { CleanUp(); }

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