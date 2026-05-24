import argparse
import hashlib
import json
import mimetypes
import re
import time
import os
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


# ============================================================
# Storage furniture ?꾩슜 ?ㅼ젙
# ============================================================
# ???뚯씪? IKEA Korea??Storage furniture 移댄뀒怨좊━留??щ·留곹빀?덈떎.
# ???援ъ“:
# - JSON:   data/raw/ikea_Storage_furniture.json
# - ?대?吏: data/images/Storage_furniture/Storage furniture_00001.jpg
# - 濡쒓렇:   data/logs/ikea_skipped_Storage_furniture.json
#
# robots.txt? ?ъ씠???댁슜 ?뺤콉???뺤씤?섍퀬, 怨쇰룄???붿껌???쇳븯湲??꾪빐
# ?붿껌 ?ъ씠??sleep???〓땲??

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
IMAGE_DIR = BASE_DIR / "data" / "images"
LOG_DIR = BASE_DIR / "data" / "logs"

TOP_CATEGORY_CODE = "storage_furniture"
TOP_CATEGORY_NAME = "Storage furniture"
START_URL = "https://www.ikea.com/kr/en/cat/storage-organisation-st001/"

# Storage furniture ?덉뿉???쒖쇅???섏쐞 移댄뀒怨좊━
EXCLUDE_CATEGORY_NAMES = {"trolleys"}

# ?ъ슜???붿껌??留욎텣 ????대쫫
OUTPUT_LABEL = "Storage_furniture"
JSON_FILE_NAME = "ikea_Storage_furniture.json"
SKIPPED_FILE_NAME = "ikea_skipped_Storage_furniture.json"
IMAGE_FOLDER_NAME = "Storage_furniture"
IMAGE_FILE_PREFIX = "Storage furniture"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

STOP_CATEGORY_TEXTS = {
    "products",
    "rooms",
    "offers",
    "planning&ideas",
    "planning & ideas",
    "services",
    "learn more",
    "recommended for you",
    "skip",
    "all services",
    "new lower price",
    "last chance to buy",
}

TEXTURE_KEYWORDS = [
    "wood",
    "veneer",
    "oak",
    "birch",
    "walnut",
    "bamboo",
    "rattan",
    "metal",
    "steel",
    "glass",
    "plastic",
    "fabric",
    "polyester",
    "cotton",
    "leather",
    "coated",
    "painted",
    "powder-coated",
    "smooth",
    "glossy",
    "matte",
    "woven",
]

FINISH_KEYWORDS = [
    "matte",
    "glossy",
    "painted",
    "coated",
    "powder-coated",
    "lacquered",
    "veneer",
    "acrylic paint",
    "foil finish",
    "clear lacquer",
    "stained",
]





COLOR_KEYWORDS = [
    "white",
    "black",
    "brown",
    "beige",
    "grey",
    "gray",
    "blue",
    "green",
    "red",
    "yellow",
    "pink",
    "orange",
    "purple",
    "transparent",
    "clear",
    "natural",
    "light",
    "dark",
    "oak",
    "birch",
    "walnut",
]
def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def safe_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_price(value):
    if value is None:
        return None
    m = re.search(r"([0-9][0-9,]*)", str(value))
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def convert_to_cm(value, unit):
    u = (unit or "cm").strip().lower()
    if u == "mm":
        return round(value / 10.0, 2)
    if u == "m":
        return round(value * 100.0, 2)
    return round(value, 2)


def parse_measure_value(text):
    if not text:
        return None
    m = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)?", str(text), re.IGNORECASE)
    if not m:
        return None
    return convert_to_cm(float(m.group("val")), m.group("unit") or "cm")


def extract_json_ld_items(soup):
    items = []
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.text
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(data, list):
            items.extend(data)
        elif isinstance(data, dict) and isinstance(data.get("@graph"), list):
            items.extend(data["@graph"])
        elif isinstance(data, dict):
            items.append(data)
    return items


def find_product_ld(items):
    for item in items:
        if not isinstance(item, dict):
            continue
        t = item.get("@type")
        if t == "Product" or (isinstance(t, list) and "Product" in t):
            return item
    return {}


def parse_dimensions(text):
    s = clean_text(text)

    def find_label_value(patterns):
        for p in patterns:
            m = re.search(p, s, re.IGNORECASE)
            if m:
                return convert_to_cm(float(m.group("val")), m.group("unit") or "cm")
        return None

    width_patterns = [
        r"(?:\bwidth\b|\b가로\b|\b폭\b)\s*[:=]?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)?",
        r"(?:^|[\s,(])W\s*[:=]\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)?",
    ]
    depth_patterns = [
        r"(?:\bdepth\b|\blength\b|\b깊이\b|\b세로\b)\s*[:=]?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)?",
        r"(?:^|[\s,(])D\s*[:=]\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)?",
    ]
    height_patterns = [
        r"(?:\bheight\b|\b높이\b|\bthickness\b)\s*[:=]?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)?",
        r"(?:^|[\s,(])H\s*[:=]\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)?",
    ]

    width_x = find_label_value(width_patterns)
    depth_y = find_label_value(depth_patterns)
    height_z = find_label_value(height_patterns)

    triple = re.search(
        r"(?P<a>\d+(?:\.\d+)?)\s*[x횞]\s*(?P<b>\d+(?:\.\d+)?)\s*[x횞]\s*(?P<c>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)",
        s,
        re.IGNORECASE,
    )
    if triple:
        unit = triple.group("unit")
        if width_x is None:
            width_x = convert_to_cm(float(triple.group("a")), unit)
        if depth_y is None:
            depth_y = convert_to_cm(float(triple.group("b")), unit)
        if height_z is None:
            height_z = convert_to_cm(float(triple.group("c")), unit)

    return width_x, depth_y, height_z


def accept_cookie_if_present(page):
    selectors = [
        "button:has-text('Accept all')",
        "button:has-text('Accept cookies')",
        "button:has-text('Allow all')",
        "button:has-text('?숈쓽')",
        "button:has-text('?덉슜')",
    ]
    for selector in selectors:
        try:
            if page.locator(selector).first.is_visible(timeout=1200):
                page.locator(selector).first.click(timeout=1200)
                return
        except Exception:
            continue


def expand_product_detail_sections(page):
    """?곹뭹 ?곸꽭???묓엺 ?곸뿭??媛?ν븳 留뚰겮 ?댁뼱 ?곸꽭 ?띿뒪?몃? ?뺣낫?⑸땲??"""
    button_texts = [
        "Product details",
        "Materials and care",
        "Measurements",
        "Good to know",
        "Designer",
        "Package details",
        "?뚯옱",
        "?쒗뭹 ?뚯옱",
        "관리",
        "移섏닔",
        "?곸꽭?뺣낫",
    ]
    for text in button_texts:
        selectors = [
            f"button:has-text('{text}')",
            f"[role='button']:has-text('{text}')",
        ]
        for selector in selectors:
            try:
                loc = page.locator(selector)
                count = min(loc.count(), 5)
                for i in range(count):
                    try:
                        item = loc.nth(i)
                        if item.is_visible(timeout=700):
                            item.click(timeout=1500)
                            time.sleep(0.2)
                    except Exception:
                        continue
            except Exception:
                continue


def save_debug_html(page_url, html):
    # debug html ?? ????
    return


def extract_category_links_from_page(page):
    entries = page.eval_on_selector_all(
        "main a[href]",
        "els => els.map(a => ({href: a.href, text: (a.innerText || a.textContent || '').trim()}))",
    )

    out = []
    for item in entries:
        href = clean_text(item.get("href"))
        text = clean_text(item.get("text"))
        if not href or "/kr/en/cat/" not in href:
            continue
        if not text:
            continue

        lowered = text.lower().replace(" ", "")
        if lowered in {x.replace(" ", "") for x in STOP_CATEGORY_TEXTS}:
            continue
        if len(text) > 80:
            continue

        clean_url = href.split("?")[0].split("#")[0]
        out.append({"url": clean_url, "name": text})

    dedup = []
    seen = set()
    for item in out:
        key = (item["url"], item["name"].lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def extract_product_links_from_page(page, limit_remaining=0):
    links = page.eval_on_selector_all("a[href]", "els => els.map(a => a.href)")
    html = page.content()
    html_links = re.findall(r"https://www\.ikea\.com/kr/en/p/[^\"'<>\s]+", html, re.IGNORECASE)

    merged = links + html_links
    out = []
    seen = set()
    for link in merged:
        clean = link.split("?")[0].split("#")[0]
        if "/kr/en/p/" not in clean:
            continue
        if clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if limit_remaining and len(out) >= limit_remaining:
            break
    return out


def is_excluded_category(name):
    return clean_text(name).lower() in EXCLUDE_CATEGORY_NAMES


def collect_category_pages(page, leaf_limit=0, max_category_pages=200):
    """
    Storage furniture ?쒖옉 URL?먯꽌 ?섏쐞 移댄뀒怨좊━瑜??곕씪 ?대젮媛묐땲??
    ?곹뭹 留곹겕媛 諛쒓껄?섎뒗 紐⑤뱺 移댄뀒怨좊━ ?섏씠吏瑜??곹뭹 紐⑸줉 ?섏씠吏 ?꾨낫濡???ν빀?덈떎.
    """
    queue = deque([(START_URL, [TOP_CATEGORY_NAME])])
    visited = set()
    product_pages = []

    while queue:
        if max_category_pages and len(visited) >= max_category_pages:
            print(f"[category stop] max_category_pages ?꾨떖: {max_category_pages}")
            break

        url, category_path = queue.popleft()
        norm_url = url.split("?")[0].split("#")[0]
        if norm_url in visited:
            continue
        visited.add(norm_url)

        print(f"[category] {' > '.join(category_path)} | {norm_url}")

        try:
            page.goto(norm_url, wait_until="domcontentloaded", timeout=90000)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except PlaywrightTimeoutError:
                pass

            accept_cookie_if_present(page)
            for _ in range(4):
                page.mouse.wheel(0, 2500)
                time.sleep(0.4)

            html = page.content()
            save_debug_html(norm_url, html)

            product_links = extract_product_links_from_page(page, limit_remaining=5)
            if product_links:
                product_pages.append({"url": norm_url, "category_path": category_path})
                print(f"  ?곹뭹 紐⑸줉 ?꾨낫 異붽?: {len(product_links)}媛??댁긽 諛쒓껄")

            child_links = extract_category_links_from_page(page)
            for child in child_links:
                child_name = clean_text(child["name"])
                child_url = child["url"]

                if is_excluded_category(child_name):
                    print(f"  ?쒖쇅 移댄뀒怨좊━ skip: {child_name}")
                    continue
                if child_name.lower() in {p.lower() for p in category_path}:
                    continue
                if child_url in visited:
                    continue

                queue.append((child_url, category_path + [child_name]))

            if leaf_limit and len(product_pages) >= leaf_limit:
                print(f"[category stop] leaf_limit ?꾨떖: {leaf_limit}")
                break

            time.sleep(0.7)

        except Exception as e:
            print(f"[category skip] {norm_url} | {e}")
            continue

    dedup = []
    seen = set()
    for item in product_pages:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        dedup.append(item)

    return dedup


def parse_middle_small(category_path):
    if len(category_path) <= 1:
        return None, None
    if len(category_path) == 2:
        return category_path[1], None
    return category_path[1], category_path[-1]


def extract_section_text(text, labels, stop_labels=None, max_lines=30):
    stop_labels = stop_labels or []
    lines = [clean_text(line) for line in str(text or "").splitlines()]
    lines = [line for line in lines if line]

    lower_labels = [label.lower() for label in labels]
    lower_stops = [label.lower() for label in stop_labels]

    for i, line in enumerate(lines):
        lower_line = line.lower()
        if any(label in lower_line for label in lower_labels):
            collected = []
            for next_line in lines[i + 1 : i + 1 + max_lines]:
                lower_next = next_line.lower()
                if collected and any(stop in lower_next for stop in lower_stops):
                    break
                collected.append(next_line)
            result = clean_text(" ".join(collected))
            return result or None
    return None


def extract_materials(text):
    return extract_section_text(
        text,
        labels=[
            "materials and care",
            "material",
            "materials",
            "소재 및 관리",
            "?쒗뭹 ?뚯옱",
            "?뚯옱",
        ],
        stop_labels=[
            "care",
            "care instructions",
            "measurements",
            "package",
            "designer",
            "good to know",
        ],
        max_lines=35,
    )


def extract_care_instructions(text):
    return extract_section_text(
        text,
        labels=["care", "care instructions", "관리", "愿由?諛⑸쾿"],
        stop_labels=["measurements", "package", "designer", "good to know", "assembly"],
        max_lines=25,
    )


def extract_finish(text):
    haystack = clean_text(text).lower()
    found = []
    for keyword in FINISH_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", haystack, re.IGNORECASE):
            found.append(keyword)
    return ", ".join(dict.fromkeys(found)) if found else None


def extract_texture_keywords(text):
    haystack = clean_text(text).lower()
    found = []
    for keyword in TEXTURE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", haystack, re.IGNORECASE):
            found.append(keyword)
    return list(dict.fromkeys(found))


def extract_color(text):
    haystack = clean_text(text).lower()
    for color in COLOR_KEYWORDS:
        if re.search(rf"\b{re.escape(color)}\b", haystack, re.IGNORECASE):
            return color
    return None


def extract_article_number(text):
    m = re.search(r"Article\s+number\s*[:\n\s]*([0-9.]+)", str(text), re.IGNORECASE)
    return clean_text(m.group(1)) if m else None


def make_image_description(product_name, category_path, width_x, depth_y, height_z, materials=None, color=None):
    path_text = " > ".join(category_path)
    parts = [f"{product_name} ?곹뭹??????대?吏?대떎."]
    parts.append(f"移댄뀒怨좊━ 寃쎈줈??{path_text}?대떎.")
    if materials:
        parts.append(f"?뚯옱 ?뺣낫??{clean_text(materials)[:150]}?대떎.")
    if color:
        parts.append(f"?됱긽 ?ㅼ썙?쒕뒗 {color}?대떎.")
    parts.append(f"?ш린??媛濡?{width_x}cm, 源딆씠 {depth_y}cm, ?믪씠 {height_z}cm?대떎.")
    return " ".join(parts)


def download_image(image_url, index):
    if not image_url:
        return "", ""
    if image_url.startswith("//"):
        image_url = "https:" + image_url

    out_dir = IMAGE_DIR / IMAGE_FOLDER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(image_url, timeout=40, headers=HEADERS)
    response.raise_for_status()

    mime = response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    ext = mimetypes.guess_extension(mime)
    if not ext:
        ext = Path(urlparse(image_url).path).suffix or ".jpg"

    file_name = f"{IMAGE_FILE_PREFIX}_{index:05d}{ext}"
    file_path = out_dir / file_name
    file_path.write_bytes(response.content)

    return str(file_path.relative_to(BASE_DIR)).replace("\\", "/"), mime


def parse_product_detail(page, product_url, category_path, index):
    page.goto(product_url, wait_until="domcontentloaded", timeout=90000)
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except PlaywrightTimeoutError:
        pass

    accept_cookie_if_present(page)

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")

    json_ld_items = extract_json_ld_items(soup)
    product_ld = find_product_ld(json_ld_items)

    product_name = clean_text(product_ld.get("name"))
    if not product_name:
        h1 = soup.find("h1")
        product_name = clean_text(h1.get_text(" ") if h1 else "")

    price = None
    offers = product_ld.get("offers")
    if isinstance(offers, dict):
        price = parse_price(offers.get("price"))
    elif isinstance(offers, list) and offers:
        price = parse_price(offers[0].get("price"))

    if price is None:
        meta_price = soup.select_one("meta[property='product:price:amount']")
        if meta_price:
            price = parse_price(meta_price.get("content"))

    body_text = page.locator("body").inner_text(timeout=30000)
    if price is None:
        price = parse_price(body_text)

    image_url = ""
    image_value = product_ld.get("image")
    if isinstance(image_value, str):
        image_url = image_value
    elif isinstance(image_value, list) and image_value:
        first_image = image_value[0]
        if isinstance(first_image, dict):
            image_url = clean_text(first_image.get("contentUrl") or first_image.get("url"))
        else:
            image_url = clean_text(first_image)
    elif isinstance(image_value, dict):
        image_url = clean_text(image_value.get("contentUrl") or image_value.get("url"))

    if not image_url:
        og = soup.select_one("meta[property='og:image']")
        if og:
            image_url = clean_text(og.get("content"))

    product_description = clean_text(product_ld.get("description"))
    if not product_description:
        meta_desc = soup.select_one("meta[name='description']") or soup.select_one("meta[property='og:description']")
        if meta_desc:
            product_description = clean_text(meta_desc.get("content"))

    width_x, depth_y, height_z = parse_dimensions(body_text)
    width_x = width_x if width_x is not None else parse_measure_value(product_ld.get("width"))
    depth_y = depth_y if depth_y is not None else parse_measure_value(product_ld.get("depth"))
    height_z = height_z if height_z is not None else parse_measure_value(product_ld.get("height"))

    if height_z is None and product_description:
        m = re.search(r"from\s*(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)", product_description, re.IGNORECASE)
        if m:
            height_z = convert_to_cm(float(m.group(2)), m.group(3))

    middle_category_name, small_category_name = parse_middle_small(category_path)

    materials = extract_materials(body_text)
    texture_source_text = " ".join(filter(None, [materials, product_description, clean_text(body_text)[:3000]]))
    texture_keywords = extract_texture_keywords(texture_source_text)
    color = extract_color(" ".join(filter(None, [product_name, product_description, clean_text(body_text)[:1500]])))

    if not all([product_name, price is not None, width_x is not None, depth_y is not None, height_z is not None, image_url]):
        return {
            "skipped": True,
            "reason": "required_field_missing",
            "product_url": product_url,
            "product_name": product_name,
            "price": price,
            "width_x": width_x,
            "depth_y": depth_y,
            "height_z": height_z,
            "image_url": image_url,
            "top_category_code": TOP_CATEGORY_CODE,
            "top_category_name": TOP_CATEGORY_NAME,
            "category_path": category_path,
            "middle_category_name": middle_category_name,
            "small_category_name": small_category_name,
            "materials": materials,
            "texture_keywords": texture_keywords,
        }

    image_path, image_mime = download_image(image_url, index)
    image_filename = Path(image_path).name if image_path else ""
    if not image_path:
        return {
            "skipped": True,
            "reason": "image_download_failed",
            "product_url": product_url,
            "product_name": product_name,
            "price": price,
            "width_x": width_x,
            "depth_y": depth_y,
            "height_z": height_z,
            "image_url": image_url,
            "top_category_code": TOP_CATEGORY_CODE,
            "top_category_name": TOP_CATEGORY_NAME,
            "category_path": category_path,
            "middle_category_name": middle_category_name,
            "small_category_name": small_category_name,
            "materials": materials,
            "texture_keywords": texture_keywords,
        }

    return {
        "top_category_code": TOP_CATEGORY_CODE,
        "top_category_name": TOP_CATEGORY_NAME,
        "category_path": category_path,
        "middle_category_name": middle_category_name,
        "small_category_name": small_category_name,
        "product_name": product_name,
        "product_url": product_url,
        "price": price,
        "width_x": width_x,
        "depth_y": depth_y,
        "height_z": height_z,
        "size_unit": "cm",
        "color": color,
        "materials": materials,
        "texture_keywords": texture_keywords,
        "image_path": image_path,
        "image_filename": image_filename,
    }


def dedupe_urls_with_path(product_url_to_path):
    return list(product_url_to_path.keys())


def crawl_storage_furniture(limit=300, leaf_limit=0, max_category_pages=200, headless=True):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    products = []
    skipped = []
    product_url_to_path = {}
    interrupted = False

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(user_agent=HEADERS["User-Agent"], viewport={"width": 1600, "height": 1200})
            page = context.new_page()

            category_pages = collect_category_pages(
                page=page,
                leaf_limit=leaf_limit,
                max_category_pages=max_category_pages,
            )

            print(f"\n?곹뭹 紐⑸줉 ?꾨낫 ?섏씠吏 ?? {len(category_pages)}\n")

            for idx, item in enumerate(category_pages, start=1):
                page_url = item["url"]
                category_path = item["category_path"]
                try:
                    page.goto(page_url, wait_until="domcontentloaded", timeout=90000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except PlaywrightTimeoutError:
                        pass

                    accept_cookie_if_present(page)
                    for _ in range(8):
                        page.mouse.wheel(0, 3500)
                        time.sleep(0.5)

                    links = extract_product_links_from_page(page, limit_remaining=0)
                    print(f"[{idx}/{len(category_pages)}] ?? ?? {len(links)}? | {' > '.join(category_path)}")

                    for link in links:
                        if link not in product_url_to_path:
                            product_url_to_path[link] = category_path

                    time.sleep(0.8)
                except Exception as e:
                    print(f"[product list skip] {page_url} | {e}")
                    skipped.append(
                        {
                            "skipped": True,
                            "reason": f"list_page_exception: {e}",
                            "page_url": page_url,
                            "category_path": category_path,
                        }
                    )

            product_urls = dedupe_urls_with_path(product_url_to_path)
            if limit:
                product_urls = product_urls[:limit]

            print(f"\n以묐났 ?쒓굅 ???곹뭹 ?섏쭛 ??? {len(product_urls)}媛?n")

            for i, product_url in enumerate(product_urls, start=1):
                category_path = product_url_to_path.get(product_url, [TOP_CATEGORY_NAME])
                print(f"[{i}/{len(product_urls)}] ?? ??: {product_url}")
                try:
                    parsed = parse_product_detail(page, product_url, category_path, i)
                    if parsed.get("skipped"):
                        skipped.append(parsed)
                    else:
                        products.append(parsed)
                except Exception as e:
                    skipped.append(
                        {
                            "skipped": True,
                            "reason": f"detail_exception: {e}",
                            "product_url": product_url,
                            "top_category_code": TOP_CATEGORY_CODE,
                            "top_category_name": TOP_CATEGORY_NAME,
                            "category_path": category_path,
                            "middle_category_name": parse_middle_small(category_path)[0],
                            "small_category_name": parse_middle_small(category_path)[1],
                        }
                    )
                time.sleep(1.0)

            context.close()
            browser.close()

    except KeyboardInterrupt:
        interrupted = True
        print("\n[?? ??] ???? ??? ??? ?????...")
    finally:
        raw_file = RAW_DIR / JSON_FILE_NAME
        skipped_file = LOG_DIR / SKIPPED_FILE_NAME

        safe_write_json(raw_file, products)
        safe_write_json(skipped_file, skipped)

        if interrupted:
            print("\n========== ?? ?? ?? ==========")
        else:
            print("\n========== ?? ==========")
        print(f"?? ?? ?: {len(products)}")
        print(f"?? ?: {len(skipped)}")
        print(f"?? JSON ??: {raw_file}")
        print(f"?? ?? ??: {skipped_file}")
        print(f"??? ??: {IMAGE_DIR / IMAGE_FOLDER_NAME}")


def main():
    parser = argparse.ArgumentParser(description="IKEA Storage furniture ?? ???")
    parser.add_argument("--limit", type=int, default=300, help="?? ?? ?? ?? ??. 0?? ?? ??")
    parser.add_argument("--leaf-limit", type=int, default=0, help="?? ?? ?? ???? ??? ??. 0?? ?? ??")
    parser.add_argument("--max-category-pages", type=int, default=200, help="??? ?? ???? ??? ?")
    parser.add_argument("--headful", action="store_true", help="???? ?? ??? ??")
    args = parser.parse_args()

    crawl_storage_furniture(
        limit=args.limit,
        leaf_limit=args.leaf_limit,
        max_category_pages=args.max_category_pages,
        headless=not args.headful,
    )


if __name__ == "__main__":
    main()



