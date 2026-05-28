# 파일 위치: /InteriorPlatform_Workspace/services/matching-engine/src/
# 파일 명: gpt_reasoning.py

import json
import os
from openai import OpenAI

# OpenAI 클라이언트 초기화 (환경 변수 또는 직접 입력)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_recommendation(query, target_id, perception_data, db_candidates):
    print(f"[LLM Engine] Starting reasoning for target_id: {target_id}")
    
    system_prompt = """
    You are an intelligent interior design AI for a 3D digital twin platform.
    Your job is to recommend the best fit furniture based on spatial data and database candidates.
    
    [CRITICAL INSTRUCTION]
    You MUST output strictly in JSON format. The Unity 3D engine will parse this directly.
    You MUST include the exact 3D filename (.ply) from the database candidates into the 'recommended_ply' key.
    If the exact .ply filename is not provided in candidates, infer a logical one (e.g., 'micke.ply' for MICKE desk).
    
    Format:
    {
        "status": "success",
        "answer": "한국어로 추천 이유, 크기, 남은 공간 등을 자세히 설명하세요.",
        "recommended_ply": "micke.ply",
        "model": "gpt-4o-mini"
    }
    """
    
    user_prompt = f"""
    - User Query: {query}
    - Target Furniture ID: {target_id}
    - Spatial Perception Data: {json.dumps(perception_data, ensure_ascii=False)}
    - Database Candidates: {json.dumps(db_candidates, ensure_ascii=False)}
    
    Based on the candidates, select the best one and return the JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"}, # 🔥 GPT가 무조건 JSON으로만 대답하게 강제
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        result_json_str = response.choices[0].message.content
        result_data = json.loads(result_json_str)
        
        # ====================================================================
        # 🔥 [파이썬 백엔드 강제 보정 로직]
        # GPT가 recommended_ply를 빼먹었거나 파일명을 이상하게 줬을 경우를 대비한 안전장치
        # ====================================================================
        if "recommended_ply" not in result_data or not result_data["recommended_ply"].endswith(".ply"):
            answer_text = result_data.get("answer", "").upper()
            
            if "MICKE" in answer_text:
                result_data["recommended_ply"] = "micke.ply"
            elif "LAGKAPTEN" in answer_text:
                result_data["recommended_ply"] = "lagkapten.ply"
            else:
                result_data["recommended_ply"] = "asset_unknown.ply"
        
        print(f"[LLM Engine] Reasoning completed successfully. Selected PLY: {result_data['recommended_ply']}")
        
        return result_data
        
    except Exception as e:
        print(f"[LLM Engine] Error during API call: {e}")
        return {
            "status": "error",
            "answer": "추천을 생성하는 중 오류가 발생했습니다.",
            "recommended_ply": "asset_unknown.ply",
            "model": "none"
        }