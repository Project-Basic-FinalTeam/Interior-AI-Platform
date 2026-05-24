import argparse
import json
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]


def get_connection():
    load_dotenv(BASE_DIR / ".env")

    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3308"))
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "1234")
    database = os.getenv("DB_NAME", "product_db")

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def load_json(json_path):
    p = Path(json_path)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 루트는 배열이어야 합니다.")
    return p, data


def resolve_image_path(image_path):
    if not image_path:
        return None
    p = Path(image_path)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def get_category_id(cur, category_code):
    cur.execute("SELECT id FROM categories WHERE category_code = %s LIMIT 1", (category_code,))
    row = cur.fetchone()
    return row["id"] if row else None


def upsert_product(cur, category_id, item):
    cur.execute(
        """
        INSERT INTO products (
            category_id, product_name, price,
            width_x, depth_y, height_z, size_unit,
            product_url, source_site, raw_size_text, raw_json
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            category_id=VALUES(category_id),
            product_name=VALUES(product_name),
            price=VALUES(price),
            width_x=VALUES(width_x),
            depth_y=VALUES(depth_y),
            height_z=VALUES(height_z),
            size_unit=VALUES(size_unit),
            source_site=VALUES(source_site),
            raw_size_text=VALUES(raw_size_text),
            raw_json=VALUES(raw_json),
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            category_id,
            item.get("product_name"),
            int(item.get("price")),
            float(item.get("width_x")),
            float(item.get("depth_y")),
            float(item.get("height_z")),
            item.get("size_unit", "cm"),
            item.get("product_url"),
            item.get("source_site", "IKEA"),
            item.get("raw_size_text"),
            json.dumps(item, ensure_ascii=False),
        ),
    )

    cur.execute("SELECT id FROM products WHERE product_url=%s LIMIT 1", (item.get("product_url"),))
    row = cur.fetchone()
    if not row:
        raise RuntimeError("product id 조회 실패")
    return row["id"]


def upsert_image(cur, product_id, item):
    image_path = resolve_image_path(item.get("image_path"))
    firebase_image_url = item.get("firebase_image_url") or item.get("image_url")
    image_filename = item.get("image_filename") or (image_path.name if image_path else None) or "firebase_image"
    image_mime = item.get("image_mime", "image/jpeg")

    image_data = b""
    if image_path and image_path.exists():
        image_data = image_path.read_bytes()

    cur.execute("DELETE FROM product_images WHERE product_id=%s", (product_id,))
    cur.execute(
        """
        INSERT INTO product_images (
            product_id, image_url, image_filename,
            image_mime, image_data, image_description, is_main
        ) VALUES (%s, %s, %s, %s, %s, %s, TRUE)
        """,
        (
            product_id,
            firebase_image_url,
            image_filename,
            image_mime,
            image_data,
            item.get("image_description"),
        ),
    )


def validate_item(item):
    required = [
        "product_name",
        "price",
        "width_x",
        "depth_y",
        "height_z",
        "top_category_code",
        "product_url",
    ]
    if not all(item.get(k) not in (None, "") for k in required):
        return False

    if not item.get("image_path") and not item.get("firebase_image_url") and not item.get("image_url"):
        return False

    return True


def insert_from_json(json_path):
    path, items = load_json(json_path)
    print(f"입력 JSON: {path}")
    print(f"읽은 상품 수: {len(items)}")

    conn = get_connection()
    success = 0
    failed = 0

    try:
        with conn.cursor() as cur:
            for idx, item in enumerate(items, start=1):
                try:
                    if not validate_item(item):
                        raise ValueError("필수 필드 누락")

                    category_code = item.get("top_category_code") or item.get("category_code")
                    category_id = get_category_id(cur, category_code)
                    if not category_id:
                        raise ValueError(f"category_code 미존재: {category_code}")

                    product_id = upsert_product(cur, category_id, item)
                    upsert_image(cur, product_id, item)

                    success += 1
                    print(f"[{idx}/{len(items)}] 저장 성공: {item.get('product_name')}")
                except Exception as e:
                    failed += 1
                    print(f"[{idx}/{len(items)}] 저장 실패: {item.get('product_url')} | {e}")

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"저장 성공 개수: {success}")
    print(f"저장 실패 개수: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="예: data/raw/ikea_Beds_mattresses.json")
    args = parser.parse_args()

    insert_from_json(args.json_path)


if __name__ == "__main__":
    main()
