// 파일 위치: /InteriorPlatform_Workspace/InteriorUnity/Assets/Scripts/Core/
// 파일 명: HandReceiver.cs

using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class HandReceiver : MonoBehaviour
{
    [Header("Target Transforms")]
    public Transform thumbTip; 
    public Transform indexTip;

    [Header("Settings")]
    [Tooltip("손가락 움직임의 부드러움 정도 (값이 클수록 반응이 빠르고 거칠어짐)")]
    public float smoothSpeed = 15f; 

    private UdpClient udpClient;
    private Thread receiveThread;
    
    // 데이터 보관용 타겟 변수
    private Vector3 _thumbTargetPos;
    private Vector3 _indexTargetPos;
    private bool _newDataReceived = false;
    private readonly object _dataLock = new object();

    void Start()
    {
        udpClient = new UdpClient(5052);
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void ReceiveData()
    {
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try {
                byte[] data = udpClient.Receive(ref anyIP);
                string received = Encoding.UTF8.GetString(data);
                
                // 데이터 파싱 및 타겟 좌표 업데이트
                string[] fingers = received.Split('|');
                if (fingers.Length == 2)
                {
                    string[] t = fingers[0].Split(',');
                    string[] i = fingers[1].Split(',');

                    lock (_dataLock)
                    {
                        _thumbTargetPos = new Vector3(float.Parse(t[0]), float.Parse(t[1]), float.Parse(t[2]));
                        _indexTargetPos = new Vector3(float.Parse(i[0]), float.Parse(i[1]), float.Parse(i[2]));
                        _newDataReceived = true;
                    }
                }
            } catch { }
        }
    }

    void Update()
    {
        if (_newDataReceived)
        {
            lock (_dataLock)
            {
                // 부드러운 움직임을 위해 Lerp 적용
                if (thumbTip != null)
                    thumbTip.position = Vector3.Lerp(thumbTip.position, _thumbTargetPos, Time.deltaTime * smoothSpeed);
                
                if (indexTip != null)
                    indexTip.position = Vector3.Lerp(indexTip.position, _indexTargetPos, Time.deltaTime * smoothSpeed);
            }
        }
    }

    void OnApplicationQuit() 
    { 
        if (receiveThread != null) receiveThread.Abort();
        if (udpClient != null) udpClient.Close(); 
    }
}