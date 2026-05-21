Shader "Custom/RuntimeGaussianSplat"
{
    Properties {
        _SplatSize ("Splat Size Multiplier", Range(0.1, 10.0)) = 1.0
    }
    SubShader
    {
        Tags { 
            "Queue"="Transparent" 
            "RenderType"="Transparent" 
            "IgnoreProjector"="True" 
            "RenderPipeline"="UniversalPipeline" 
        }
        ZWrite Off
        Cull Off 
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            Name "Forward"
            Tags { "LightMode"="UniversalForward" }

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            struct SplatData {
                float3 pos;
                float3 color;
                float opacity;
                float3 scale;
                float4 rot; // w, x, y, z
            };

            StructuredBuffer<SplatData> _SplatBuffer;
            float _SplatSize;
            float4x4 _CustomLocalToWorld;

            struct appdata {
                float4 vertex : POSITION;
                uint instanceID : SV_InstanceID;
            };

            struct v2f {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float2 uv : TEXCOORD0;
            };

            // 🔥 [회전 복원] 쿼터니언을 이용한 벡터 회전 함수 (이게 없어서 모양이 다 뭉개진 겁니다)
            float3 rotateVector(float3 v, float4 q) {
                return v + 2.0 * cross(q.yzw, cross(q.yzw, v) + q.x * v);
            }

            v2f vert (appdata v)
            {
                v2f o;
                SplatData splat = _SplatBuffer[v.instanceID];

                // 1. 월드 좌표 배치
                float3 centerWorld = mul(_CustomLocalToWorld, float4(splat.pos, 1.0)).xyz;

                // 2. 스케일 적용
                float3 trueScale = exp(splat.scale) * _SplatSize;
                
                // 3. 🔥 [핵심] 가구 고유 회전 적용 (쿼터니언)
                float3 localPos = v.vertex.xyz * trueScale;
                localPos = rotateVector(localPos, splat.rot);
                
                // 4. 빌보드 (카메라 정면 정렬)
                float3 camRight = float3(UNITY_MATRIX_V._m00, UNITY_MATRIX_V._m01, UNITY_MATRIX_V._m02);
                float3 camUp    = float3(UNITY_MATRIX_V._m10, UNITY_MATRIX_V._m11, UNITY_MATRIX_V._m12);
                float3 worldPos = centerWorld + (camRight * localPos.x + camUp * localPos.y);
                
                o.pos = mul(UNITY_MATRIX_VP, float4(worldPos, 1.0));
                
                // 5. 색상 및 투명도
                float3 rgb = max(0, splat.color * 0.28209 + 0.5);
                float alpha = 1.0 / (1.0 + exp(-splat.opacity));

                o.color = float4(rgb, alpha);
                o.uv = v.vertex.xy;
                
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float r2 = dot(i.uv, i.uv);
                if (r2 > 1.0) discard;
                float gaussianAlpha = exp(-r2 * 4.0) * i.color.a;
                return float4(i.color.rgb, gaussianAlpha);
            }
            ENDCG
        }
    }
}