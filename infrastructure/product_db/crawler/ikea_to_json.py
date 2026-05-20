import argparse
import json
import mimetypes
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
IMAGE_DIR = BASE_DIR / "data" / "images"
LOG_DIR = BASE_DIR / "data" / "logs"


CATEGORY_URLS = {
    "desk": {
        "category_name": "책상",
        "url": "https://www.ikea.com/kr/en/cat/desks-computer-desks-20649/",
    },
    "chair": {
        "category_name": "의자",
        "url": "https://www.ikea.com/kr/en/cat/chairs-fu002/",
    },
    "shelf": {
        "category_name": "선반",
        "url": "https://www.ikea.com/kr/en/cat/bookcases-shelving-units-st002/",
    },
}


def clean_text(text):
    if not text:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def parse_price(text):
    if text is None:
        return None

    matches = re.findall(r"\d[\d,]*", str(text))

    if not matches:
        return None

    return int(matches[0].replace(",", ""))


def convert_to_cm(value, unit):
    unit = (unit or "cm").lower()

    if unit == "mm":
        return value / 10

    if unit == "m":
        return value * 100

    return value


def find_dimension(text, labels):
    for label in labels:
        pattern = rf"{label}\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|m)?"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = float(match.group(1))
            unit = match.group(2) or "cm"
            return round(convert_to_cm(value, unit), 2)

    return None


def parse_dimensions(body_text):
    text = clean_text(body_text)

    width_x = find_dimension(text, ["width", "w", "가로", "폭"])
    depth_y = find_dimension(text, ["depth", "length", "d", "깊이", "세로", "길이"])
    height_z = find_dimension(text, ["height", "h", "높이"])

    if width_x is None or depth_y is None or height_z is None:
        generic = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|m)?",
            text,
            re.IGNORECASE,
        )

        if generic:
            unit = generic.group(4) or "cm"

            if width_x is None:
                width_x = round(convert_to_cm(float(generic.group(1)), unit), 2)

            if depth_y is None:
                depth_y = round(convert_to_cm(float(generic.group(2)), unit), 2)

            if height_z is None:
                height_z = round(convert_to_cm(float(generic.group(3)), unit), 2)

    return width_x, depth_y, height_z


def extract_json_ld(soup):
    items = []

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string

        if not raw:
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(data, list):
            items.extend(data)
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                items.extend(data["@graph"])
            else:
                items.append(data)

    return items


def find_product_ld(json_ld_items):
    for item in json_ld_items:
        item_type = item.get("@type")

        if item_type == "Product":
            return item

        if isinstance(item_type, list) and "Product" in item_type:
            return item

    return {}


def get_meta_content(soup, key):
    tag = soup.find("meta", attrs={"property": key})

    if tag and tag.get("content"):
        return tag["content"]

    tag = soup.find("meta", attrs={"name": key})

    if tag and tag.get("content"):
        return tag["content"]

    return ""


def extract_product_links(page, category_url, limit):
    page.goto(category_url, wait_until="domcontentloaded", timeout=60000)

    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass

    for _ in range(6):
        page.mouse.wheel(0, 3000)
        time.sleep(1)

    links = page.eval_on_selector_all(
        "a[href]",
        "elements => elements.map(a => a.href)"
    )

    product_links = []
    seen = set()

    for link in links:
        if "/p/" not in link:
            continue

        clean_link = link.split("?")[0].split("#")[0]

        if clean_link in seen:
            continue

        seen.add(clean_link)
        product_links.append(clean_link)

        if len(product_links) >= limit:
            break

    return product_links


def download_image(image_url, category_code, index):
    if not image_url:
        return "", ""

    if image_url.startswith("//"):
        image_url = "https:" + image_url

    image_dir = IMAGE_DIR / category_code
    image_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        image_url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()

    image_mime = response.headers.get("Content-Type", "image/jpeg").split(";")[0]
    ext = mimetypes.guess_extension(image_mime)

    if not ext:
        parsed_url = urlparse(image_url)
        suffix = Path(parsed_url.path).suffix
        ext = suffix if suffix else ".jpg"

    filename = f"{category_code}_{index:04d}{ext}"
    image_path = image_dir / filename

    with open(image_path, "wb") as file:
        file.write(response.content)

    return str(image_path.relative_to(BASE_DIR)), image_mime


def make_image_description(product_name, category_name, width_x, depth_y, height_z):
    return (
        f"{product_name} 상품의 대표 이미지이다. "
        f"카테고리는 {category_name}이며, "
        f"크기는 가로 {width_x}cm, 깊이 {depth_y}cm, 높이 {height_z}cm이다."
    )


def parse_product_detail(page, product_url, category_code, category_name, index):
    page.goto(product_url, wait_until="domcontentloaded", timeout=60000)

    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")

    json_ld_items = extract_json_ld(soup)
    product_ld = find_product_ld(json_ld_items)

    product_name = clean_text(product_ld.get("name")) if product_ld else ""

    if not product_name:
        h1 = soup.find("h1")
        product_name = clean_text(h1.get_text(" ")) if h1 else ""

    price = None

    offers = product_ld.get("offers") if product_ld else None

    if isinstance(offers, dict):
        price = parse_price(offers.get("price"))

    if price is None:
        price = parse_price(get_meta_content(soup, "product:price:amount"))

    body_text = page.locator("body").inner_text(timeout=30000)

    if price is None:
        price = parse_price(body_text)

    image_url = ""

    image_value = product_ld.get("image") if product_ld else ""

    if isinstance(image_value, str):
        image_url = image_value
    elif isinstance(image_value, list) and image_value:
        image_url = image_value[0]

    if not image_url:
        image_url = get_meta_content(soup, "og:image")

    width_x, depth_y, height_z = parse_dimensions(body_text)

    if not product_name or price is None or width_x is None or depth_y is None or height_z is None:
        return {
            "skipped": True,
            "reason": "required_field_missing",
            "product_url": product_url,
            "product_name": product_name,
            "price": price,
            "width_x": width_x,
            "depth_y": depth_y,
            "height_z": height_z,
        }

    image_path, image_mime = download_image(image_url, category_code, index)

    if not image_path:
        return {
            "skipped": True,
            "reason": "image_download_failed",
            "product_url": product_url,
        }

    return {
        "category_code": category_code,
        "category_name": category_name,
        "product_name": product_name,
        "price": price,
        "width_x": width_x,
        "depth_y": depth_y,
        "height_z": height_z,
        "size_unit": "cm",
        "product_url": product_url,
        "image_url": image_url,
        "image_path": image_path,
        "image_mime": image_mime,
        "image_description": make_image_description(
            product_name,
            category_name,
            width_x,
            depth_y,
            height_z,
        ),
        "raw_size_text": body_text[:1500],
        "source_site": "IKEA",
    }


def crawl_ikea(category_code, limit):
    if category_code not in CATEGORY_URLS:
        raise ValueError(f"지원하지 않는 카테고리입니다: {category_code}")

    category = CATEGORY_URLS[category_code]
    category_name = category["category_name"]
    category_url = category["url"]

    products = []
    skipped = []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)

        page = browser.new_page(
            user_agent="Mozilla/5.0",
            viewport={"width": 1440, "height": 1000}
        )

        product_links = extract_product_links(page, category_url, limit)

        print(f"수집 대상 상품 수: {len(product_links)}")

        for index, product_url in enumerate(product_links, start=1):
            print(f"[{index}/{len(product_links)}] {product_url}")

            try:
                product = parse_product_detail(
                    page=page,
                    product_url=product_url,
                    category_code=category_code,
                    category_name=category_name,
                    index=index,
                )

                if product.get("skipped"):
                    skipped.append(product)
                else:
                    products.append(product)

            except Exception as error:
                skipped.append({
                    "skipped": True,
                    "reason": str(error),
                    "product_url": product_url,
                })

            time.sleep(1.5)

        browser.close()

    output_path = RAW_DIR / f"ikea_products_{category_code}.json"
    skipped_path = LOG_DIR / f"ikea_skipped_{category_code}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(products, file, ensure_ascii=False, indent=2)

    with open(skipped_path, "w", encoding="utf-8") as file:
        json.dump(skipped, file, ensure_ascii=False, indent=2)

    print(f"JSON 저장 완료: {output_path}")
    print(f"스킵 로그 저장 완료: {skipped_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="desk", help="desk, chair, shelf")
    parser.add_argument("--limit", type=int, default=3)

    args = parser.parse_args()

    crawl_ikea(args.category, args.limit)


if __name__ == "__main__":
    main()