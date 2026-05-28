# 파일 위치: /InteriorPlatform_Workspace/services/matching-engine/src/
# 파일 명: db_matcher.py

def find_best_candidates(category, max_width, max_height):
    print(f"[DB Matcher] Searching candidates for category: {category}")
    
    # 실제 환경에서는 PostgreSQL DB 쿼리가 들어갈 자리입니다.
    # 지금은 테스트가 바로 가능하도록 Mock 데이터를 반환합니다.
    
    mock_db_results = [
        {
            "product_id": 1,
            "name": "MICKE desk",
            "size": "105x50x72cm",
            "style": "modern",
            # 🔥 gpt_reasoning.py 및 실제 저장된 파일명과 100% 동일하게 맞춥니다.
            "ply_filename": "micke.ply" 
        },
        {
            "product_id": 2,
            "name": "LAGKAPTEN desk",
            "size": "120x60x73cm",
            "style": "minimalist",
            # 🔥 마찬가지로 파일명 통일!
            "ply_filename": "lagkapten.ply"
        }
    ]
    
    print(f"[DB Matcher] Found {len(mock_db_results)} candidates.")
    return mock_db_results