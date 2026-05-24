# Product DB (IKEA Crawling)

## 1) 준비
```powershell
cd infrastructure/product_db
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## 2) IKEA 크롤링 실행
```powershell
python .\crawler\ikea_to_json.py --category desk --limit 3
```

생성 파일:
- `data/raw/ikea_products_desk.json`
- `data/logs/ikea_skipped_desk.json`
- `data/logs/debug_category_page.html`
- `data/images/desk/*`

## 3) 크롤링 결과 확인
```powershell
Get-Content -Raw .\data\raw\ikea_products_desk.json
Get-Content -Raw .\data\logs\ikea_skipped_desk.json
```

## 4) JSON -> MySQL INSERT
기본 연결값:
- host: `127.0.0.1`
- port: `3308`
- user: `root`
- password: `1234`
- db: `product_db`

필요 시 `.env`로 오버라이드:
- `DB_HOST`
- `DB_PORT`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`

실행:
```powershell
python .\db\insert_from_json.py data/raw/ikea_products_desk.json
```

## 5) DB 저장 검증
```powershell
docker exec -i product-db-mysql mysql -uroot -p1234 product_db -e "SELECT id, product_name, price, width_x, depth_y, height_z FROM products;"
docker exec -i product-db-mysql mysql -uroot -p1234 product_db -e "SELECT product_id, image_filename, image_mime, LENGTH(image_data) AS image_size FROM product_images;"
```

## 참고
- 크롤러는 카테고리 페이지 HTML을 `data/logs/debug_category_page.html`로 저장합니다.
- 링크 수집 결과(전체 a 태그 수/상품 링크 수/링크 목록), 성공/스킵 개수를 터미널에 출력합니다.
- 치수 단위는 `mm`, `cm`, `m`를 받아 DB 저장 전 `cm`로 통일합니다.
