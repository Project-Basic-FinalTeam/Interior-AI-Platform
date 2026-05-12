// Assets/Scripts/Core/SimplePlyLoader.cs
using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Text;

public class SimplePlyLoader : MonoBehaviour
{
    public static Mesh LoadPly(string filePath)
    {
        if (!File.Exists(filePath)) return null;

        List<Vector3> vertices = new List<Vector3>();
        List<int> indices = new List<int>();
        
        using (BinaryReader reader = new BinaryReader(File.Open(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)))
        {
            int vertexCount = 0;
            int bytesPerVertex = 0;
            string line;
            bool inVertexElement = false;

            while ((line = ReadLine(reader)) != "end_header")
            {
                if (line.StartsWith("element vertex"))
                {
                    vertexCount = int.Parse(line.Split(' ')[2]);
                    inVertexElement = true;
                }
                else if (line.StartsWith("property") && inVertexElement)
                {
                    bytesPerVertex += 4; 
                }
                else if (line.StartsWith("element") && !line.StartsWith("element vertex"))
                {
                    inVertexElement = false;
                }
            }

            int maxVertices = Mathf.Min(vertexCount, 65000); 
            int step = Mathf.Max(1, vertexCount / maxVertices); 

            for (int i = 0; i < vertexCount; i++)
            {
                if (i % step == 0 && vertices.Count < 65000)
                {
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    
                    vertices.Add(new Vector3(x, -y, z)); 
                    indices.Add(vertices.Count - 1);
                    
                    reader.BaseStream.Seek(bytesPerVertex - 12, SeekOrigin.Current); 
                }
                else
                {
                    reader.BaseStream.Seek(bytesPerVertex, SeekOrigin.Current);
                }
            }
        }

        Mesh mesh = new Mesh();
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.SetVertices(vertices);
        mesh.SetIndices(indices.ToArray(), MeshTopology.Points, 0); 

        return mesh;
    }

    private static string ReadLine(BinaryReader reader)
    {
        StringBuilder sb = new StringBuilder();
        char c;
        while ((c = reader.ReadChar()) != '\n')
        {
            if (c != '\r') sb.Append(c);
        }
        return sb.ToString();
    }
}