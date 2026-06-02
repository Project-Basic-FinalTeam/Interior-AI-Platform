# 파일 위치: /InteriorPlatform_Workspace/services/matching-engine/main.py

from fastapi import FastAPI, Request
import uvicorn
import json

# 팀원분이 완벽하게 짜둔 gpt_reasoning 로직 그대로 가져오기
from src.gpt_reasoning import generate_recommendation

app = FastAPI(title="Interior AI Matching Engine")

# 임시 가구 DB (실제 DB 연동 전 테스트용. db_matcher.py가 있다면 그걸로 교체하시면 됩니다)
DUMMY_FURNITURE_DB = [
    {"id": 101, "name": "MICKE desk", "style": "modern", "color": "white", "filename": "micke.ply", "size": "105x50cm"},
    {"id": 102, "name": "LAGKAPTEN desk", "style": "minimalist", "color": "black", "filename": "lagkapten.ply", "size": "120x60cm"}
]

@app.post("/api/recommend")
async def get_recommendation(request: Request):
    try:
        # 1. 유니티(또는 C++)에서 보낸 JSON 데이터 받기
        payload = await request.json()
        query = payload.get("query", "")
        target_id = payload.get("target_id", 0)
        perception_data = payload.get("perception_data", {})

        print(f"\n[Matching API] 📩 유니티로부터 추천 요청 수신")
        print(f"- Query: {query}")
        print(f"- Target ID: {target_id}")

        # 2. 팀원분이 만든 완벽한 GPT 추론 함수 실행 (RAG)
        # db_candidates에 DUMMY_DB를 넘겨서 GPT가 고르게 만듭니다.
        result_json = generate_recommendation(
            query=query,
            target_id=target_id,
            perception_data=perception_data,
            db_candidates=DUMMY_FURNITURE_DB
        )

        # 3. 유니티로 결과 리턴
        return result_json

    except Exception as e:
        print(f"[Matching API] 🚨 오류 발생: {e}")
        return {"status": "error", "answer": "서버 내부 오류 발생", "recommended_ply": "asset_unknown.ply"}

if __name__ == "__main__":
    # 포트는 팀 규칙에 맞게 변경하세요. (ngrok이 이 포트를 물고 외부로 포워딩합니다)
    uvicorn.run(app, host="0.0.0.0", port=8001)