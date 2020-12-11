

/*----- Depth texture and motion vectors post-process shader-----
Encodes the per-pixel motion information into the r and g channels,
and the depth texture information into the b channel.
*/


Shader "Custom/DepthAndMotionVectors" {
	SubShader{
	Tags { "RenderType" = "Opaque" }

		Pass{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#include "UnityCG.cginc"

			sampler2D _CameraDepthTexture;
			sampler2D _CameraMotionVectorsTexture;

			struct v2f {
			   float4 pos : SV_POSITION;
			   float4 scrPos:TEXCOORD1;
			};

			//Vertex Shader
			v2f vert(appdata_base v) {
			   v2f o;

			   //pass screen pos to fragment shader
			   o.pos = UnityObjectToClipPos(v.vertex);
			   o.scrPos = ComputeScreenPos(o.pos);
			   return o;
			}

			//Fragment Shader
			half4 frag(v2f i) : COLOR
			{
				half4 final;
	
				//UV motion in r and g (-1 to 1 range)
				float motion_scale = 5.0;
				final.r = tex2Dproj(_CameraMotionVectorsTexture, UNITY_PROJ_COORD(i.scrPos)).r;
				final.g = tex2Dproj(_CameraMotionVectorsTexture, UNITY_PROJ_COORD(i.scrPos)).g;
				final.b = tex2Dproj(_CameraDepthTexture, UNITY_PROJ_COORD(i.scrPos)).r; //non-linear!


				//set alpha channel to 1.0
				final.a = 1.0;


				return final;
			}
		ENDCG
		}
	}
		FallBack "Diffuse"
}