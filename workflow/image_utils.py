import asyncio
import base64
import html as _html
import io
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import aiohttp
from PIL import Image

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    from astrbot.api.star import StarTools
except Exception:  # pragma: no cover
    StarTools = None  # type: ignore


_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.I)
_IMG_CLASS_RE = re.compile(r'\sclass=(?P<q>["\'])(?P<cls>[^"\']*)(?P=q)', re.I)
_FETCH_WARNED: Set[Tuple[str, int]] = set()
_INLINE_NOOP_WARNED: Set[str] = set()


def get_plugin_data_dir(subdir: str) -> Path:
    if StarTools is not None:
        try:
            return Path(StarTools.get_data_dir()) / subdir
        except Exception:
            pass
    return Path(__file__).resolve().parent.parent / "data" / subdir


def _normalize_urls(urls: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _split_url_input(value: str) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[\s\r\n]+", value.strip())
    return [p for p in (x.strip() for x in parts) if p]


def _guess_referer(url: str) -> Optional[str]:
    u = url.lower()
    if "mp.weixin.qq.com" in u or "mmbiz.qpic.cn" in u:
        return "https://mp.weixin.qq.com/"
    # MiYoShe / miHoYo image CDNs often require a referer under *.miyoushe.com or bbs.mihoyo.com
    if (
        "miyoushe.com" in u
        or "mihoyo.com" in u
        or "miyoushe.net" in u
    ):
        return "https://www.miyoushe.com/"
    return None


async def _fetch_http_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    headers = {"User-Agent": "AstrBotDailyNews/1.0"}
    referer = _guess_referer(url)
    if referer:
        headers["Referer"] = referer
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                try:
                    host = (urlparse(url).netloc or "").lower()
                    key = (host, int(resp.status))
                    if host and key not in _FETCH_WARNED:
                        _FETCH_WARNED.add(key)
                        astrbot_logger.warning(
                            "[dailynews] fetch image failed: host=%s status=%s referer=%s url=%s",
                            host,
                            resp.status,
                            referer or "",
                            url[:240],
                        )
                except Exception:
                    pass
                return None
            return await resp.read()
    except Exception:
        return None


def _decode_base64_image(url: str) -> Optional[bytes]:
    if url.startswith("base64://"):
        try:
            return base64.b64decode(url.removeprefix("base64://"))
        except Exception:
            return None
    if _DATA_URI_RE.match(url):
        try:
            return base64.b64decode(url.split(",", 1)[1])
        except Exception:
            return None
    return None


def _read_local_image(url: str) -> Optional[bytes]:
    path = url
    if url.startswith("file:///"):
        path = url.replace("file:///", "")
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None


async def fetch_images_bytes(urls: Iterable[str]) -> List[bytes]:
    normalized = _normalize_urls(urls)
    if not normalized:
        return []

    out: List[bytes] = []
    async with aiohttp.ClientSession() as session:
        for url in normalized:
            data = _decode_base64_image(url)
            if data:
                out.append(data)
                continue

            if url.startswith("http://") or url.startswith("https://"):
                data = await _fetch_http_image(session, url)
                if data:
                    out.append(data)
                continue

            data = _read_local_image(url)
            if data:
                out.append(data)

    return out


async def fetch_image_bytes_one(url: str) -> Optional[bytes]:
    data = await fetch_images_bytes([url])
    return data[0] if data else None


def probe_image_size(image_bytes: bytes) -> Optional[Tuple[int, int]]:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return int(img.width), int(img.height)
    except Exception:
        return None


async def probe_image_size_from_url(url: str) -> Optional[Tuple[int, int]]:
    b = await fetch_image_bytes_one(url)
    if not b:
        return None
    return probe_image_size(b)


async def download_image_to_jpeg_file(
    url: str,
    *,
    out_path: Path,
    max_width: int = 1200,
    quality: int = 88,
) -> Optional[Tuple[Path, Tuple[int, int]]]:
    """
    Download/convert an image URL into a local JPEG file.

    Returns (path, original_size). The file may be resized to max_width.
    """
    raw = _decode_base64_image(url)
    if raw is None:
        if url.startswith("http://") or url.startswith("https://"):
            raw = await fetch_image_bytes_one(url)
        else:
            raw = _read_local_image(url)
    if not raw:
        return None

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        orig = (int(img.width), int(img.height))
        if max_width > 0 and img.width > int(max_width):
            ratio = int(max_width) / float(img.width)
            new_h = max(1, int(img.height * ratio))
            img = img.resize((int(max_width), new_h), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=int(quality), optimize=True)
        return out_path, orig
    except Exception:
        return None


def _to_jpeg_bytes(data: bytes, *, max_width: int = 1200, quality: int = 85) -> Optional[bytes]:
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        if img.width > max_width:
            ratio = max_width / float(img.width)
            new_h = max(1, int(img.height * ratio))
            img = img.resize((max_width, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
        return buf.getvalue()
    except Exception:
        return None


async def image_url_to_data_uri(url: str) -> Optional[str]:
    raw = await fetch_image_bytes_one(url)
    if not raw:
        return None
    jpeg = _to_jpeg_bytes(raw)
    if not jpeg:
        return None
    b64 = base64.b64encode(jpeg).decode("ascii")
    return "data:image/jpeg;base64," + b64


_IMG_SRC_RE = re.compile(r"""<img(?P<before>[^>]*?)\ssrc=(?P<q>["'])(?P<src>[^"']+)(?P=q)(?P<after>[^>]*)>""", re.I)


async def inline_html_remote_images(
    html: str,
    *,
    max_images: int = 8,
    max_width: int = 1200,
    quality: int = 85,
) -> str:
    """
    将 HTML 中的远程图片（http/https）内联成 data-uri，避免防盗链/渲染端无法带 Referer。
    只处理 <img src="...">，并限制数量以避免生成过大的 HTML。
    """
    s = html or ""
    if not s:
        return s

    srcs: List[str] = []
    for m in _IMG_SRC_RE.finditer(s):
        src = _html.unescape((m.group("src") or "").strip())
        if src.startswith("http://") or src.startswith("https://"):
            if src not in srcs:
                srcs.append(src)
        if len(srcs) >= max_images:
            break

    if not srcs:
        return s

    mapping: Dict[str, str] = {}
    for src in srcs:
        data = await fetch_image_bytes_one(src)
        if not data:
            continue
        jpeg = _to_jpeg_bytes(data, max_width=max_width, quality=quality)
        if not jpeg:
            continue
        mapping[src] = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

    if not mapping:
        try:
            host = (urlparse(srcs[0]).netloc or "").lower() if srcs else ""
            if host and host not in _INLINE_NOOP_WARNED:
                _INLINE_NOOP_WARNED.add(host)
                astrbot_logger.warning(
                    "[dailynews] inline_html_remote_images: failed to inline any images (count=%s, host=%s)",
                    len(srcs),
                    host,
                )
        except Exception:
            pass
        return s

    def repl(m: re.Match) -> str:
        src_raw = (m.group("src") or "").strip()
        src = _html.unescape(src_raw)
        new_src = mapping.get(src)
        if not new_src:
            return m.group(0)
        q = m.group("q")
        before = m.group("before") or ""
        after = m.group("after") or ""
        return f"<img{before} src={q}{new_src}{q}{after}>"

    return _IMG_SRC_RE.sub(repl, s)


async def localize_html_remote_images(
    html: str,
    *,
    max_images: int = 8,
    max_width: int = 1200,
    quality: int = 85,
) -> str:
    """
    Replace remote <img src="http(s)://..."> with local file:// URLs.

    This avoids embedding huge base64 data-uri in HTML (which may flood logs),
    while still bypassing common hotlink/referer restrictions by downloading locally.
    """
    s = html or ""
    if not s:
        return s

    srcs: List[str] = []
    for m in _IMG_SRC_RE.finditer(s):
        src = _html.unescape((m.group("src") or "").strip())
        if src.startswith("http://") or src.startswith("https://"):
            if src not in srcs:
                srcs.append(src)
        if len(srcs) >= max_images:
            break

    if not srcs:
        return s

    import hashlib

    out_dir = get_plugin_data_dir("html_image_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, str] = {}
    for src in srcs:
        try:
            h = hashlib.sha1(src.encode("utf-8", errors="ignore")).hexdigest()[:20]
            out_path = out_dir / f"{h}.jpg"
            saved = await download_image_to_jpeg_file(
                src,
                out_path=out_path,
                max_width=max_width,
                quality=quality,
            )
            if not saved:
                continue
            saved_path, _ = saved
            mapping[src] = f"file:///{saved_path.resolve().as_posix()}"
        except Exception:
            continue

    if not mapping:
        return s

    def repl(m: re.Match) -> str:
        src_raw = (m.group("src") or "").strip()
        src = _html.unescape(src_raw)
        new_src = mapping.get(src)
        if not new_src:
            return m.group(0)
        q = m.group("q")
        before = m.group("before") or ""
        after = m.group("after") or ""
        return f"<img{before} src={q}{new_src}{q}{after}>"

    return _IMG_SRC_RE.sub(repl, s)


def _classify_image_for_layout(
    *,
    width: int,
    height: int,
    full_max_width: int,
    medium_max_width: int,
    narrow_max_width: int,
    float_if_width_le: int,
    float_enabled: bool,
    float_dir: str,
) -> List[str]:
    classes = ["md-img"]
    if width <= 0 or height <= 0:
        classes.append("md-img--full")
        return classes

    aspect = width / float(height)
    if aspect >= 1.75:
        classes.append("md-img--panorama")
    elif aspect >= 1.15:
        classes.append("md-img--landscape")
    elif aspect >= 0.9:
        classes.append("md-img--square")
    else:
        classes.append("md-img--portrait")

    resized_w = min(width, full_max_width)
    if resized_w <= float_if_width_le or aspect <= 0.92 or width <= narrow_max_width:
        classes.append("md-img--narrow")
        if float_enabled:
            classes.append("md-img--float-r" if float_dir == "r" else "md-img--float-l")
        return classes

    if resized_w <= medium_max_width:
        classes.append("md-img--medium")
        return classes

    classes.append("md-img--full")
    return classes


def _merge_img_class_attr(tag: str, classes_to_add: List[str]) -> str:
    add = " ".join([c for c in classes_to_add if c])
    if not add:
        return tag
    m = _IMG_CLASS_RE.search(tag)
    if not m:
        return tag.replace("<img", f'<img class="{add}"', 1)
    existing = (m.group("cls") or "").strip()
    merged = (existing + " " + add).strip() if existing else add
    return tag[: m.start("cls")] + merged + tag[m.end("cls") :]


def _inject_img_data_attrs(tag: str, *, width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return tag
    insert = f' data-w="{int(width)}" data-h="{int(height)}"'
    if tag.startswith("<img") and " data-w=" not in tag:
        return tag.replace("<img", "<img" + insert, 1)
    return tag


async def adaptive_layout_html_images(
    html: str,
    *,
    max_images: int = 12,
    full_max_width: int = 1000,
    medium_max_width: int = 820,
    narrow_max_width: int = 420,
    float_if_width_le: int = 480,
    float_enabled: bool = True,
) -> str:
    """
    Post-process HTML to standardize image display size and prefer text-side-by-side
    for images that are small after normalization (via float classes).

    The HTML is expected to come from markdown_to_html() (no raw HTML from users).
    """
    s = html or ""
    if not s:
        return s

    matches = list(_IMG_SRC_RE.finditer(s))
    if not matches:
        return s

    # Collect unique srcs in order, capped.
    srcs: List[str] = []
    for m in matches:
        src = _html.unescape((m.group("src") or "").strip())
        if not src:
            continue
        if src not in srcs:
            srcs.append(src)
        if len(srcs) >= max_images:
            break

    if not srcs:
        return s

    sizes: dict[str, Tuple[int, int]] = {}
    for src in srcs:
        b = _decode_base64_image(src)
        if b is None and (src.startswith("http://") or src.startswith("https://")):
            b = await fetch_image_bytes_one(src)
        if b is None:
            b = _read_local_image(src)
        if not b:
            continue
        sz = probe_image_size(b)
        if sz:
            sizes[src] = sz

    float_toggle = "r"

    def repl(m: re.Match) -> str:
        src_raw = (m.group("src") or "").strip()
        src = _html.unescape(src_raw)
        before = m.group("before") or ""
        after = m.group("after") or ""
        q = m.group("q")
        tag = f"<img{before} src={q}{src}{q}{after}>"
        sz = sizes.get(src)
        nonlocal float_toggle
        if sz:
            w, h = sz
            classes = _classify_image_for_layout(
                width=w,
                height=h,
                full_max_width=full_max_width,
                medium_max_width=medium_max_width,
                narrow_max_width=narrow_max_width,
                float_if_width_le=float_if_width_le,
                float_enabled=float_enabled,
                float_dir=float_toggle,
            )
            if "md-img--float-r" in classes or "md-img--float-l" in classes:
                float_toggle = "l" if float_toggle == "r" else "r"
            tag = _merge_img_class_attr(tag, classes)
            tag = _inject_img_data_attrs(tag, width=w, height=h)
        else:
            tag = _merge_img_class_attr(tag, ["md-img", "md-img--full"])
        return tag

    out = _IMG_SRC_RE.sub(repl, s, count=max_images)

    # Mark image-only paragraphs to fine-tune spacing in CSS.
    out = re.sub(
        r"<p>\s*(<img[^>]*>)\s*</p>",
        r'<p class="md-imgp">\1</p>',
        out,
        flags=re.I,
    )

    # If a floated narrow image is immediately followed by a heading/divider (which clears floats),
    # it will create a large empty column; de-float in that case.
    def _defloat(m: re.Match) -> str:
        img = m.group("img") or ""
        nxt = m.group("next") or ""
        img = re.sub(r"\bmd-img--float-(?:r|l)\b", "", img)
        img = re.sub(r"\s{2,}", " ", img)
        return f'<p class="md-imgp">{img}</p>{nxt}'

    out = re.sub(
        r'(?is)<p class="md-imgp">\s*(?P<img><img[^>]*\bmd-img--float-(?:r|l)\b[^>]*>)\s*</p>'
        r'(?P<next>\s*(?:<div class="md-heading\b|<div class="css-snow-divider\b|<h[1-6]\b|<hr\b))',
        _defloat,
        out,
    )
    return out

def _merge_images_vertical_sync(
    images_bytes: List[bytes],
    out_path: Path,
    *,
    max_width: int = 1080,
    gap: int = 8,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Path:
    pil_images: List[Image.Image] = []
    widths: List[int] = []
    for b in images_bytes:
        try:
            img = Image.open(io.BytesIO(b))
            img = img.convert("RGB")
            pil_images.append(img)
            widths.append(img.width)
        except Exception:
            continue

    if not pil_images:
        raise RuntimeError("no valid images to merge")

    target_width = min(max(widths), max_width)
    resized: List[Image.Image] = []
    total_height = 0
    for img in pil_images:
        if img.width > target_width:
            ratio = target_width / float(img.width)
            new_h = max(1, int(img.height * ratio))
            img = img.resize((target_width, new_h), Image.LANCZOS)
        if img.width < target_width:
            canvas = Image.new("RGB", (target_width, img.height), bg_color)
            x = (target_width - img.width) // 2
            canvas.paste(img, (x, 0))
            img = canvas
        resized.append(img)
        total_height += img.height

    total_height += gap * (len(resized) - 1)
    merged = Image.new("RGB", (target_width, total_height), bg_color)
    y = 0
    for idx, img in enumerate(resized):
        merged.paste(img, (0, y))
        y += img.height
        if idx != len(resized) - 1:
            y += gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path, format="JPEG", quality=92)
    return out_path


async def merge_images_vertical(
    urls: Iterable[str],
    *,
    out_path: Path,
    max_width: int = 1080,
    gap: int = 8,
) -> Path:
    images_bytes = await fetch_images_bytes(urls)
    if not images_bytes:
        raise RuntimeError("no images downloaded")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _merge_images_vertical_sync(
            images_bytes,
            out_path,
            max_width=max_width,
            gap=gap,
        ),
    )


def parse_image_urls(value: object) -> List[str]:
    if isinstance(value, list):
        return _normalize_urls([str(x) for x in value])
    if isinstance(value, str):
        return _normalize_urls(_split_url_input(value))
    return []
