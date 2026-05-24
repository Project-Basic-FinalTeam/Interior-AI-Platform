import argparse
import hashlib
import json
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, storage


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "data" / "logs"

DEFAULT_CATEGORIES = [
    "Storage_furniture",
    "Storage_accessories",
    "Beds_mattresses",
    "Sofas_armchairs",
    "Tables_chairs",
    "Desks_office_chairs",
]


def load_env():
    load_dotenv(BASE_DIR / ".env")


def init_firebase():
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT", "").strip()
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "").strip()

    if not service_account_path:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT 값이 없습니다.")
    if not bucket_name:
        raise ValueError("FIREBASE_STORAGE_BUCKET 값이 없습니다.")

    cred_path = Path(service_account_path)
    if not cred_path.is_absolute():
        cred_path = BASE_DIR / cred_path
    if not cred_path.exists():
        raise FileNotFoundError(f"서비스 계정 키 파일이 없습니다: {cred_path}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

    return firestore.client(), storage.bucket()


def load_products(category_label):
    json_path = RAW_DIR / f"ikea_{category_label}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"입력 JSON 파일이 없습니다: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"invalid_json: 루트가 배열이 아님 ({json_path})")
    return data, json_path


def make_doc_id(item, category_label, index):
    base = (
        item.get("product_url")
        or f"{item.get('top_category_name')}_{item.get('product_name')}_{item.get('image_filename')}"
        or f"{category_label}_{index}_{item.get('product_name')}"
    )
    return hashlib.sha1(str(base).encode("utf-8")).hexdigest()


def make_storage_path(category_label, image_filename):
    prefix = os.getenv("FIREBASE_STORAGE_PREFIX", "furniture_images")
    return f"{prefix}/{category_label}/{image_filename}"


def make_download_url(bucket_name, storage_path, token):
    encoded_path = quote(storage_path, safe="")
    return f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media&token={token}"


def upload_image(bucket, item, category_label, skip_existing_storage=False):
    image_path = item.get("image_path")
    image_filename = item.get("image_filename")

    if not image_path or not image_filename:
        raise FileNotFoundError("image_file_missing: image_path 또는 image_filename 없음")

    local_path = Path(image_path)
    if not local_path.is_absolute():
        local_path = BASE_DIR / local_path

    if not local_path.exists():
        raise FileNotFoundError(f"image_file_missing: {local_path}")

    storage_path = make_storage_path(category_label, image_filename)
    blob = bucket.blob(storage_path)

    if skip_existing_storage and blob.exists():
        return {
            "firebase_storage_path": storage_path,
            "firebase_gs_url": f"gs://{bucket.name}/{storage_path}",
            "firebase_image_url": None,
        }

    content_type, _ = mimetypes.guess_type(str(local_path))
    if content_type is None:
        content_type = "image/jpeg"

    download_token = str(uuid.uuid4())
    blob.metadata = {"firebaseStorageDownloadTokens": download_token}
    blob.upload_from_filename(str(local_path), content_type=content_type)

    return {
        "firebase_storage_path": storage_path,
        "firebase_gs_url": f"gs://{bucket.name}/{storage_path}",
        "firebase_image_url": make_download_url(bucket.name, storage_path, download_token),
    }


def build_firestore_item(item, upload_result, category_label):
    now = datetime.now(timezone.utc).isoformat()

    result = dict(item)
    result.update(upload_result)
    result["source_site"] = "IKEA KR English"
    result["source_category_label"] = category_label
    result["firebase_uploaded"] = True
    result["imported_at"] = now
    return result


def save_outputs(category_label, success_items, failed_items):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    processed_path = PROCESSED_DIR / f"ikea_{category_label}_firebase.json"
    failed_path = LOG_DIR / f"firebase_import_failed_{category_label}.json"

    with processed_path.open("w", encoding="utf-8") as f:
        json.dump(success_items, f, ensure_ascii=False, indent=2)

    with failed_path.open("w", encoding="utf-8") as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2)

    print(f"[OUTPUT] {processed_path}")
    print(f"[FAILED_LOG] {failed_path}")


def import_category(category_label, limit=0, dry_run=False, overwrite=True, skip_existing_storage=False):
    db = None
    bucket = None
    if not dry_run:
        db, bucket = init_firebase()

    collection_name = os.getenv("FIREBASE_COLLECTION", "ikea_products")
    products, input_path = load_products(category_label)

    if limit and limit > 0:
        products = products[:limit]

    print(f"[START] category={category_label}")
    print(f"[INPUT] {input_path}")
    print(f"[COUNT] total={len(products)}, limit={limit}")

    success_items = []
    failed_items = []

    batch = db.batch() if not dry_run else None
    batch_count = 0

    try:
        for index, item in enumerate(products, start=1):
            try:
                doc_id = make_doc_id(item, category_label, index)
                image_filename = item.get("image_filename")
                storage_path = make_storage_path(category_label, image_filename or f"{index}.jpg")

                if dry_run:
                    local = Path(item.get("image_path", ""))
                    if not local.is_absolute():
                        local = BASE_DIR / local
                    print(f"[DRY-RUN] image exists: {local.exists()} path={local}")
                    print(f"[DRY-RUN] storage path: {storage_path}")
                    print(f"[DRY-RUN] firestore doc_id: {doc_id}")
                    continue

                upload_result = upload_image(bucket, item, category_label, skip_existing_storage=skip_existing_storage)
                firestore_item = build_firestore_item(item, upload_result, category_label)

                doc_ref = db.collection(collection_name).document(doc_id)
                if overwrite:
                    batch.set(doc_ref, firestore_item, merge=True)
                else:
                    batch.create(doc_ref, firestore_item)
                batch_count += 1
                success_items.append(firestore_item)

                print(f"[UPLOAD] {index}/{len(products)} {item.get('image_filename')} -> {upload_result['firebase_storage_path']}")
                print(f"[FIRESTORE] saved doc_id={doc_id}")

                if batch_count >= 400:
                    batch.commit()
                    print("[COMMIT] 400 documents committed")
                    batch = db.batch()
                    batch_count = 0

            except Exception as e:
                failed_items.append(
                    {
                        "skipped": True,
                        "reason": f"unexpected_exception:{type(e).__name__}:{e}",
                        "category_label": category_label,
                        "index": index,
                        "product_name": item.get("product_name"),
                        "image_path": item.get("image_path"),
                        "image_filename": item.get("image_filename"),
                        "product_url": item.get("product_url"),
                    }
                )

    except KeyboardInterrupt:
        print("[INTERRUPTED] Ctrl+C detected. Saving current progress...")
    finally:
        if not dry_run and batch_count > 0:
            batch.commit()
            print("[COMMIT] final batch committed")

        if not dry_run:
            save_outputs(category_label, success_items, failed_items)

    print(f"[DONE] success={len(success_items)}, failed={len(failed_items)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing-storage", action="store_true")
    args = parser.parse_args()

    load_env()

    if not args.category and not args.all:
        raise ValueError("--category 또는 --all 중 하나는 필요합니다.")

    if args.all:
        for category in DEFAULT_CATEGORIES:
            json_path = RAW_DIR / f"ikea_{category}.json"
            if not json_path.exists():
                print(f"[SKIP] input not found: {json_path}")
                continue
            import_category(
                category_label=category,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite or True,
                skip_existing_storage=args.skip_existing_storage,
            )
    else:
        import_category(
            category_label=args.category,
            limit=args.limit,
            dry_run=args.dry_run,
            overwrite=args.overwrite or True,
            skip_existing_storage=args.skip_existing_storage,
        )


if __name__ == "__main__":
    main()
