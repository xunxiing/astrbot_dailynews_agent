from __future__ import annotations

import base64
import io
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ..core.config_models import RenderImageStyleConfig
from ..core.image_utils import fetch_image_bytes_one, get_plugin_data_dir

_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.I)
_WS_RE = re.compile(r"[ \t\r\f\v]+")


@dataclass(frozen=True)
class InlineToken:
    text: str
    style: str = "body"


@dataclass(frozen=True)
class FloatBox:
    side: str
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class Theme:
    name: str
    width: int
    hero_h: int
    body_font: int
    h1: int
    h2: int
    h3: int
    title: int
    subtitle: int
    brand: int
    code: int
    page_bg: tuple[int, int, int, int]
    paper_bg: tuple[int, int, int, int]
    paper_border: tuple[int, int, int, int]
    text: tuple[int, int, int, int]
    muted: tuple[int, int, int, int]
    accent: tuple[int, int, int, int]
    accent_soft: tuple[int, int, int, int]
    quote_bg: tuple[int, int, int, int]
    quote_border: tuple[int, int, int, int]
    code_bg: tuple[int, int, int, int]
    code_head: tuple[int, int, int, int]
    link: tuple[int, int, int, int]


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_font_path() -> Path:
    return (_plugin_root() / "font" / "HYWenHei-75W-2.ttf").resolve(strict=False)


def _theme(is_chenyu: bool) -> Theme:
    if is_chenyu:
        return Theme(
            name="chenyu",
            width=1280,
            hero_h=280,
            body_font=30,
            h1=34,
            h2=28,
            h3=24,
            title=60,
            subtitle=24,
            brand=28,
            code=18,
            page_bg=(240, 244, 240, 255),
            paper_bg=(255, 255, 255, 148),
            paper_border=(45, 90, 39, 120),
            text=(24, 40, 28, 255),
            muted=(31, 79, 36, 230),
            accent=(45, 90, 39, 255),
            accent_soft=(45, 90, 39, 150),
            quote_bg=(244, 248, 244, 220),
            quote_border=(45, 90, 39, 90),
            code_bg=(248, 250, 252, 236),
            code_head=(100, 116, 139, 255),
            link=(31, 79, 36, 255),
        )
    return Theme(
        name="daily",
        width=1120,
        hero_h=260,
        body_font=23,
        h1=28,
        h2=28,
        h3=24,
        title=46,
        subtitle=18,
        brand=12,
        code=16,
        page_bg=(255, 255, 255, 255),
        paper_bg=(255, 255, 255, 218),
        paper_border=(17, 24, 39, 26),
        text=(17, 24, 39, 255),
        muted=(75, 85, 99, 255),
        accent=(37, 99, 235, 255),
        accent_soft=(90, 140, 245, 180),
        quote_bg=(241, 245, 249, 235),
        quote_border=(148, 163, 184, 160),
        code_bg=(248, 250, 252, 245),
        code_head=(100, 116, 139, 255),
        link=(37, 99, 235, 255),
    )


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").replace("\xa0", " "))


def _decode_image_bytes(value: str) -> bytes | None:
    s = str(value or "").strip()
    if not s:
        return None
    if _DATA_URI_RE.match(s):
        try:
            return base64.b64decode(s.split(",", 1)[1])
        except Exception:
            return None
    if len(s) > 256 and "/" not in s and "\\" not in s:
        try:
            return base64.b64decode(s)
        except Exception:
            return None
    return None


def _image_from_bytes(data: bytes | None) -> Image.Image | None:
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        return None


def _fit_cover(
    dst: Image.Image,
    src: Image.Image | None,
    box: tuple[int, int, int, int],
    opacity: float = 1.0,
) -> None:
    if src is None:
        return
    x0, y0, x1, y1 = box
    img = ImageOps.fit(
        src.convert("RGBA"), (max(1, x1 - x0), max(1, y1 - y0)), method=Image.LANCZOS
    )
    if opacity < 1.0:
        alpha = img.getchannel("A").point(lambda x: int(x * opacity))
        img.putalpha(alpha)
    dst.alpha_composite(img, (x0, y0))


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask


def _draw_glass_panel(
    dst: Image.Image,
    *,
    box: tuple[int, int, int, int],
    radius: int,
    blur_radius: float,
    overlay_fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    outline_width: int,
) -> None:
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    region = dst.crop((x0, y0, x1, y1)).convert("RGBA")
    blurred = region.filter(ImageFilter.GaussianBlur(radius=max(0.1, float(blur_radius))))
    mask = _rounded_mask((width, height), int(radius))
    dst.paste(blurred, (x0, y0), mask)

    overlay = Image.new("RGBA", (width, height), overlay_fill)
    dst.paste(overlay, (x0, y0), mask)

    draw = ImageDraw.Draw(dst)
    draw.rounded_rectangle(box, radius=radius, outline=outline, width=int(outline_width))


class HtmlToPillow:
    def __init__(
        self,
        *,
        width: int,
        theme: Theme,
        style: RenderImageStyleConfig,
        image_cache: dict[str, Image.Image | None],
    ):
        self.width = width
        self.theme = theme
        self.style = style
        self.image_cache = image_cache
        self.font_path = _default_font_path()
        self.canvas = Image.new("RGBA", (self.width, 2400), (255, 255, 255, 0))
        self.draw = ImageDraw.Draw(self.canvas)
        self.y = 0
        self.float_box: FloatBox | None = None
        self.font_cache: dict[int, Any] = {}

    def _font(self, size: int) -> Any:
        size = int(size)
        if size in self.font_cache:
            return self.font_cache[size]
        try:
            font = ImageFont.truetype(str(self.font_path), size)
        except Exception:
            font = ImageFont.load_default()
        self.font_cache[size] = font
        return font

    def _ensure(self, height: int) -> None:
        if height <= self.canvas.height:
            return
        new_h = max(self.canvas.height * 2, height + 1024)
        out = Image.new("RGBA", (self.width, new_h), (255, 255, 255, 0))
        out.alpha_composite(self.canvas)
        self.canvas = out
        self.draw = ImageDraw.Draw(self.canvas)

    def _measure(self, text: str, size: int) -> tuple[int, int]:
        box = self.draw.textbbox((0, 0), text, font=self._font(size))
        return max(0, int(box[2] - box[0])), max(0, int(box[3] - box[1]))

    def _bounds(self, indent: int = 0) -> tuple[int, int]:
        if self.float_box and self.y >= self.float_box.bottom:
            self.float_box = None
        left = indent
        right = self.width
        if self.float_box and self.y < self.float_box.bottom:
            if self.float_box.side == "right":
                right = min(right, self.float_box.x - 16)
            else:
                left = max(left, self.float_box.right + 16)
            if right - left < 180:
                self.finish_float()
                return self._bounds(indent)
        return left, right

    def finish_float(self) -> None:
        if self.float_box is None:
            return
        self.y = max(self.y, self.float_box.bottom + 8)
        self.float_box = None

    async def _load_image(self, src: str) -> Image.Image | None:
        ref = str(src or "").strip()
        if not ref:
            return None
        if ref in self.image_cache:
            cached = self.image_cache.get(ref)
            return cached.copy() if cached is not None else None
        data = _decode_image_bytes(ref)
        if data is None:
            try:
                data = await fetch_image_bytes_one(ref)
            except Exception:
                data = None
        img = _image_from_bytes(data)
        self.image_cache[ref] = img.copy() if img is not None else None
        return img

    def _tokens(self, node: Any, style: str = "body") -> list[InlineToken]:
        if isinstance(node, NavigableString):
            txt = _normalize_text(str(node))
            return [InlineToken(txt, style)] if txt else []
        if not isinstance(node, Tag):
            return []
        if node.name == "br":
            return [InlineToken("\n", style)]
        if node.name == "img":
            return []
        next_style = style
        if node.name in {"strong", "b"}:
            next_style = "strong"
        elif node.name in {"em", "i"}:
            next_style = "em"
        elif node.name == "code":
            next_style = "code"
        elif node.name == "a":
            next_style = "link"
        out: list[InlineToken] = []
        for child in node.children:
            out.extend(self._tokens(child, next_style))
        return out

    def _draw_rich(
        self,
        tokens: list[InlineToken],
        *,
        size: int,
        color: tuple[int, int, int, int],
        indent: int = 0,
        first_prefix: str = "",
        gap: float = 1.5,
        margin: int = 8,
    ) -> None:
        chars: list[tuple[str, str]] = []
        for tok in tokens:
            if tok.text == "\n":
                chars.append(("\n", tok.style))
            else:
                chars.extend((ch, tok.style) for ch in tok.text)
        if not chars:
            return
        idx = 0
        first = True
        while idx < len(chars):
            left, right = self._bounds(indent)
            prefix = first_prefix if first else ""
            prefix_w = self._measure(prefix, size)[0] if prefix else 0
            limit = max(60, right - left - prefix_w)
            line: list[tuple[str, str]] = []
            width = 0
            while idx < len(chars):
                ch, st = chars[idx]
                if ch == "\n":
                    idx += 1
                    break
                if not line and ch.isspace():
                    idx += 1
                    continue
                cw = self._measure(ch, max(14, size - 2) if st == "code" else size)[0]
                if line and width + cw > limit:
                    break
                line.append((ch, st))
                width += cw
                idx += 1
            if not line and idx < len(chars):
                line.append(chars[idx])
                idx += 1
            self._ensure(self.y + size * 3)
            cursor_x = left
            if prefix:
                self.draw.text(
                    (cursor_x, self.y),
                    prefix,
                    font=self._font(size),
                    fill=self.theme.accent,
                )
                cursor_x += prefix_w
            line_h = 0
            buf = ""
            cur_style = ""
            groups: list[tuple[str, str]] = []
            for ch, st in line:
                if not buf or st == cur_style:
                    buf += ch
                    cur_style = st
                else:
                    groups.append((buf, cur_style))
                    buf = ch
                    cur_style = st
            if buf:
                groups.append((buf, cur_style))
            for text, st in groups:
                seg_size = max(14, size - 2) if st == "code" else size
                fill = (
                    self.theme.link
                    if st == "link"
                    else self.theme.accent
                    if st == "strong"
                    else self.theme.muted
                    if st == "em"
                    else color
                )
                tw, th = self._measure(text, seg_size)
                if st == "code" and tw > 0:
                    self.draw.rounded_rectangle(
                        (cursor_x - 2, self.y - 1, cursor_x + tw + 4, self.y + th + 3),
                        radius=6,
                        fill=(235, 240, 245, 255),
                    )
                self.draw.text(
                    (cursor_x, self.y), text, font=self._font(seg_size), fill=fill
                )
                if st == "link" and tw > 0:
                    self.draw.line(
                        (cursor_x, self.y + th + 1, cursor_x + tw, self.y + th + 1),
                        fill=fill,
                        width=1,
                    )
                cursor_x += tw
                line_h = max(line_h, th)
            self.y += max(line_h + 2, int(math.ceil(line_h * gap)))
            first = False
        self.y += margin

    async def _render_img(self, tag: Tag) -> None:
        src = str(tag.get("src") or "").strip()
        img = await self._load_image(src)
        if img is None:
            return
        cls = {str(x) for x in (tag.get("class") or [])}
        data_w = int(tag.get("data-w") or img.width)
        data_h = int(tag.get("data-h") or img.height)
        aspect = data_w / float(max(1, data_h))
        target_w = min(self.width, data_w)
        if "md-img--full" in cls:
            target_w = min(self.width, int(self.style.full_max_width))
        elif "md-img--medium" in cls:
            target_w = min(self.width, int(self.style.medium_max_width))
        elif "md-img--narrow" in cls:
            target_w = min(self.width, int(self.style.narrow_max_width))
        target_h = max(1, int(target_w / max(0.05, aspect)))
        if "md-img--portrait" in cls and target_h > 420:
            target_h = 420
            target_w = max(1, int(target_h * aspect))
        if "md-img--panorama" in cls and target_h > 320:
            target_h = 320
            target_w = max(1, int(target_h * aspect))
        img = img.resize((target_w, target_h), Image.LANCZOS)
        side = (
            "right"
            if "md-img--float-r" in cls
            else "left"
            if "md-img--float-l" in cls
            else ""
        )
        if side:
            self.finish_float()
            left, right = self._bounds(0)
            x = right - img.width if side == "right" else left
            y = self.y + 4
            self._ensure(y + img.height + 20)
            self.canvas.alpha_composite(img, (x, y))
            self.draw.rounded_rectangle(
                (x, y, x + img.width, y + img.height),
                radius=10,
                outline=(148, 163, 184, 110),
                width=1,
            )
            self.float_box = FloatBox(side, x, y, img.width, img.height)
            return
        self.finish_float()
        self._ensure(self.y + img.height + 20)
        x = max(0, (self.width - img.width) // 2)
        y = self.y + 6
        self.canvas.alpha_composite(img, (x, y))
        self.draw.rounded_rectangle(
            (x, y, x + img.width, y + img.height),
            radius=10,
            outline=(148, 163, 184, 110),
            width=1,
        )
        self.y = y + img.height + 16

    async def _render_p(self, tag: Tag) -> None:
        imgs = [c for c in tag.children if isinstance(c, Tag) and c.name == "img"]
        if imgs and not _normalize_text(tag.get_text(" ", strip=True).replace(" ", "")):
            for img in imgs:
                await self._render_img(img)
            return
        tokens = self._tokens(tag)
        if tokens:
            self._draw_rich(tokens, size=self.theme.body_font, color=self.theme.text)
        for img in imgs:
            await self._render_img(img)

    async def _render_list(self, tag: Tag, depth: int = 0) -> None:
        ordered = tag.name == "ol"
        items = [c for c in tag.children if isinstance(c, Tag) and c.name == "li"]
        for idx, li in enumerate(items, start=1):
            holder = BeautifulSoup("<div></div>", "html.parser").div
            nested: list[Tag] = []
            for child in li.children:
                if isinstance(child, Tag) and child.name in {"ul", "ol"}:
                    nested.append(child)
                else:
                    holder.append(
                        child if not isinstance(child, NavigableString) else str(child)
                    )
            bullet = f"{idx}. " if ordered else "• "
            self._draw_rich(
                self._tokens(holder),
                size=self.theme.body_font,
                color=self.theme.text,
                indent=depth * 34,
                first_prefix=bullet,
                gap=1.45,
                margin=4,
            )
            for sub in nested:
                await self._render_list(sub, depth + 1)
        self.y += 4

    async def _render_blockquote(self, tag: Tag) -> None:
        self.finish_float()
        sub = HtmlToPillow(
            width=max(220, self.width - 44),
            theme=self.theme,
            style=self.style,
            image_cache=self.image_cache,
        )
        await sub.render_children(tag.children)
        box = sub.to_image()
        y = self.y + 6
        self._ensure(y + box.height + 40)
        self.draw.rounded_rectangle(
            (0, y, self.width, y + box.height + 28),
            radius=14,
            fill=self.theme.quote_bg,
            outline=self.theme.quote_border,
            width=2,
        )
        self.canvas.alpha_composite(box, (22, y + 14))
        self.y = y + box.height + 38

    def _wrap_plain(self, text: str, size: int, width: int) -> list[str]:
        lines: list[str] = []
        cur = ""
        for ch in str(text or ""):
            if ch == "\n":
                lines.append(cur.rstrip())
                cur = ""
                continue
            if not cur and ch.isspace():
                continue
            test = cur + ch
            if cur and self._measure(test, size)[0] > width:
                lines.append(cur.rstrip())
                cur = "" if ch.isspace() else ch
            else:
                cur = test
        if cur:
            lines.append(cur.rstrip())
        return lines or [""]

    async def _render_codebox(self, tag: Tag) -> None:
        self.finish_float()
        lang = (
            _normalize_text(
                (tag.find(class_="codebox-lang") or tag).get_text(" ", strip=True)
            )
            or "code"
        )
        body = tag.find(class_="codebox-body")
        code_node = body.find("code") if body is not None else tag.find("code")
        code = (
            code_node.get_text("", strip=False)
            if code_node is not None
            else body.get_text("", strip=False)
            if body is not None
            else tag.get_text("", strip=False)
        )
        code = str(code or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = self._wrap_plain(code, self.theme.code, max(160, self.width - 28))
        line_h = self._measure("Ag", self.theme.code)[1] + 6
        total_h = 44 + 20 + len(lines) * line_h
        y = self.y + 8
        self._ensure(y + total_h + 24)
        self.draw.rounded_rectangle(
            (0, y, self.width, y + total_h),
            radius=12,
            fill=self.theme.code_bg,
            outline=(148, 163, 184, 140),
            width=1,
        )
        self.draw.rounded_rectangle(
            (0, y, self.width, y + 44), radius=12, fill=self.theme.code_head
        )
        self.draw.text(
            (16, y + 10), lang, font=self._font(16), fill=(248, 250, 252, 255)
        )
        cy = y + 54
        for line in lines:
            self.draw.text(
                (14, cy), line, font=self._font(self.theme.code), fill=self.theme.text
            )
            cy += line_h
        self.y = y + total_h + 10

    async def _render_table(self, tag: Tag) -> None:
        self.finish_float()
        rows = []
        for tr in tag.find_all("tr"):
            rows.append(
                [
                    _normalize_text(c.get_text(" ", strip=True))
                    for c in tr.find_all(["th", "td"], recursive=False)
                ]
            )
        if not rows:
            return
        cols = max(len(r) for r in rows)
        rows = [r + [""] * (cols - len(r)) for r in rows]
        col_w = max(120, self.width // max(1, cols))
        line_h = self._measure("Ag", max(16, self.theme.body_font - 4))[1] + 8
        wrapped = []
        heights = []
        for row in rows:
            lines_row = [
                self._wrap_plain(cell, max(16, self.theme.body_font - 4), col_w - 24)
                for cell in row
            ]
            wrapped.append(lines_row)
            heights.append(max(len(x) for x in lines_row) * line_h + 18)
        y = self.y + 8
        total_h = sum(heights)
        self._ensure(y + total_h + 24)
        self.draw.rounded_rectangle(
            (0, y, self.width, y + total_h),
            radius=12,
            fill=(255, 255, 255, 220),
            outline=(45, 90, 39, 60),
            width=1,
        )
        cy = y
        for ridx, row in enumerate(wrapped):
            rh = heights[ridx]
            if ridx == 0:
                self.draw.rectangle(
                    (0, cy, self.width, cy + rh), fill=(230, 238, 230, 220)
                )
            elif ridx % 2 == 1:
                self.draw.rectangle(
                    (0, cy, self.width, cy + rh), fill=(245, 249, 245, 180)
                )
            for cidx, lines in enumerate(row):
                cx = cidx * col_w
                self.draw.rectangle(
                    (cx, cy, cx + col_w, cy + rh), outline=(45, 90, 39, 40), width=1
                )
                ty = cy + 10
                for line in lines:
                    self.draw.text(
                        (cx + 12, ty),
                        line,
                        font=self._font(max(16, self.theme.body_font - 4)),
                        fill=self.theme.text,
                    )
                    ty += line_h
            cy += rh
        self.y = y + total_h + 10

    def _divider(self) -> None:
        self.finish_float()
        self._ensure(self.y + 36)
        y = self.y + 12
        tri_w = 26
        self.draw.line(
            (0, y, self.width - tri_w - 10, y), fill=self.theme.accent_soft, width=3
        )
        cx = self.width - tri_w - 2
        self.draw.polygon(
            [(cx, y + 8), (cx + tri_w // 2, y - 6), (cx + tri_w, y + 8)],
            fill=self.theme.accent,
        )
        self.y += 26

    async def _render_node(self, node: Any) -> None:
        if isinstance(node, NavigableString):
            if not _normalize_text(str(node)).strip():
                return
            p = BeautifulSoup("<p></p>", "html.parser").p
            p.append(str(node))
            await self._render_p(p)
            return
        if not isinstance(node, Tag):
            return
        cls = {str(x) for x in (node.get("class") or [])}
        if node.name == "div" and "section-box" in cls:
            inner = node.find(class_="md") or node
            await self.render_children(inner.children)
            return
        if node.name == "div" and "md" in cls:
            await self.render_children(node.children)
            return
        if node.name == "div" and "md-heading" in cls:
            h = node.find(["h2", "h3"])
            if h is not None:
                await self._render_heading(h, 2 if h.name == "h2" else 3)
            return
        if node.name == "div" and "css-snow-divider" in cls:
            self._divider()
            return
        if node.name == "div" and "codebox" in cls:
            await self._render_codebox(node)
            return
        if node.name == "h1":
            await self._render_heading(node, 1)
            return
        if node.name == "h2":
            await self._render_heading(node, 2)
            return
        if node.name == "h3":
            await self._render_heading(node, 3)
            return
        if node.name == "p":
            await self._render_p(node)
            return
        if node.name in {"ul", "ol"}:
            await self._render_list(node)
            return
        if node.name == "blockquote":
            await self._render_blockquote(node)
            return
        if node.name == "table":
            await self._render_table(node)
            return
        if node.name == "pre":
            await self._render_codebox(node)
            return
        if node.name == "img":
            await self._render_img(node)
            return
        if node.name == "hr":
            self._divider()
            return
        await self.render_children(node.children)

    async def _render_heading(self, tag: Tag, level: int) -> None:
        self.finish_float()
        size = (
            self.theme.h1
            if level == 1
            else self.theme.h2
            if level == 2
            else self.theme.h3
        )
        color = self.theme.text if level == 1 else self.theme.accent
        self._draw_rich(
            self._tokens(tag),
            size=size,
            color=color,
            gap=1.25,
            margin=10 if level == 1 else 6,
        )

    async def render_children(self, children: Any) -> None:
        for child in list(children):
            await self._render_node(child)

    def to_image(self) -> Image.Image:
        self.finish_float()
        h = max(1, min(self.canvas.height, self.y + 4))
        return self.canvas.crop((0, 0, self.width, h))


async def _render_body(
    body_html: str,
    width: int,
    theme: Theme,
    style: RenderImageStyleConfig,
    image_cache: dict[str, Image.Image | None],
) -> Image.Image:
    soup = BeautifulSoup(f'<div class="md-root">{body_html}</div>', "html.parser")
    root = soup.find(class_="md-root")
    renderer = HtmlToPillow(
        width=width, theme=theme, style=style, image_cache=image_cache
    )
    renderer.y = 6
    if root is not None:
        await renderer.render_children(root.children)
    return renderer.to_image()


def _sections(body_html: str) -> list[str]:
    soup = BeautifulSoup(f'<div class="md-root">{body_html}</div>', "html.parser")
    root = soup.find(class_="md-root")
    if root is None:
        return []
    boxes = root.find_all(class_="section-box", recursive=False)
    if not boxes:
        return [body_html]
    out = []
    for box in boxes:
        inner = box.find(class_="md") or box
        out.append("".join(str(x) for x in inner.contents))
    return out


def _gradient(width: int, height: int, start_alpha: int, end_alpha: int) -> Image.Image:
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        alpha = int(
            start_alpha + (end_alpha - start_alpha) * (y / float(max(1, height - 1)))
        )
        draw.line((0, y, width, y), fill=(255, 255, 255, alpha), width=1)
    return img


async def render_pillow_fallback_image(
    *,
    body_html: str,
    title: str,
    subtitle: str,
    style: RenderImageStyleConfig,
    is_chenyu: bool,
    bg_img_b64: str = "",
    char_img_b64: str = "",
    chenyu_bg_top: str = "",
    chenyu_bg_middle: str = "",
    chenyu_bg_bottom: str = "",
    chenyu_tower: str = "",
) -> Path | None:
    try:
        theme = _theme(is_chenyu)
        cache: dict[str, Image.Image | None] = {}
        title_font = ImageFont.truetype(str(_default_font_path()), theme.title)
        subtitle_font = ImageFont.truetype(str(_default_font_path()), theme.subtitle)
        brand_font = ImageFont.truetype(str(_default_font_path()), theme.brand)
        char_img = (
            _image_from_bytes(
                _decode_image_bytes("data:image/png;base64," + char_img_b64)
            )
            if char_img_b64
            else None
        )

        if is_chenyu:
            section_imgs = [
                await _render_body(sec, theme.width - 160, theme, style, cache)
                for sec in _sections(body_html)
            ]
            final_h = (
                theme.hero_h
                + 34
                + sum(img.height + 24 for img in section_imgs)
                + max(0, len(section_imgs) - 1) * 20
                + 40
            )
            canvas = Image.new(
                "RGBA", (theme.width, max(theme.hero_h + 80, final_h)), theme.page_bg
            )
            _fit_cover(
                canvas,
                _image_from_bytes(_decode_image_bytes(chenyu_bg_top)),
                (0, 0, theme.width, min(canvas.height, 600)),
                0.8,
            )
            mid = _image_from_bytes(_decode_image_bytes(chenyu_bg_middle))
            if mid is not None:
                tile = ImageOps.fit(
                    mid,
                    (
                        theme.width,
                        max(1, min(800, mid.height * theme.width // max(1, mid.width))),
                    ),
                    method=Image.LANCZOS,
                )
                alpha = tile.getchannel("A").point(lambda x: int(x * 0.7))
                tile.putalpha(alpha)
                y = min(canvas.height, 500)
                end_y = max(y, canvas.height - 700)
                while y < end_y:
                    canvas.alpha_composite(tile, (0, y))
                    y += tile.height
            _fit_cover(
                canvas,
                _image_from_bytes(_decode_image_bytes(chenyu_bg_bottom)),
                (0, max(0, canvas.height - 800), theme.width, canvas.height),
                0.72,
            )
            canvas.alpha_composite(_gradient(theme.width, theme.hero_h, 0, 110), (0, 0))
            if char_img is not None:
                h = 340
                w = max(1, int(char_img.width * (h / float(max(1, char_img.height)))))
                canvas.alpha_composite(
                    char_img.resize((w, h), Image.LANCZOS), (theme.width - w - 40, 0)
                )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (60, 42),
                "ASTRBOT DAILY NEWS",
                font=brand_font,
                fill=(255, 255, 255, 246),
            )
            draw.rectangle((60, 102, 72, 174), fill=(195, 59, 42, 255))
            draw.text((88, 94), title, font=title_font, fill=(16, 28, 18, 255))
            draw.rounded_rectangle(
                (92, 184, 420, 228), radius=6, fill=(32, 68, 35, 150)
            )
            draw.text(
                (106, 191), subtitle, font=subtitle_font, fill=(255, 255, 255, 255)
            )
            y = theme.hero_h + 20
            for sec in section_imgs:
                _draw_glass_panel(
                    canvas,
                    box=(40, y, theme.width - 40, y + sec.height + 24),
                    radius=14,
                    blur_radius=10,
                    overlay_fill=theme.paper_bg,
                    outline=theme.paper_border,
                    outline_width=2,
                )
                canvas.alpha_composite(sec, (60, y + 12))
                y += sec.height + 44
            out = canvas.crop((0, 0, theme.width, min(canvas.height, y + 4)))
        else:
            body = await _render_body(body_html, theme.width - 100, theme, style, cache)
            final_h = theme.hero_h + 18 + body.height + 92
            canvas = Image.new(
                "RGBA", (theme.width, max(theme.hero_h + 80, final_h)), theme.page_bg
            )
            _fit_cover(
                canvas,
                _image_from_bytes(
                    _decode_image_bytes("data:image/jpeg;base64," + bg_img_b64)
                )
                if bg_img_b64
                else None,
                (0, 0, theme.width, theme.hero_h),
                1.0,
            )
            canvas.alpha_composite(
                _gradient(theme.width, theme.hero_h, 26, 230), (0, 0)
            )
            if char_img is not None:
                h = 320
                w = max(1, int(char_img.width * (h / float(max(1, char_img.height)))))
                canvas.alpha_composite(
                    char_img.resize((w, h), Image.LANCZOS), (theme.width - w - 16, 8)
                )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (30, 22),
                "ASTRBOT DAILY NEWS",
                font=brand_font,
                fill=(255, 255, 255, 235),
            )
            draw.rectangle((30, 74, 38, 130), fill=(192, 57, 43, 255))
            draw.text((54, 66), title, font=title_font, fill=(15, 23, 42, 255))
            draw.text((54, 128), subtitle, font=subtitle_font, fill=theme.muted)
            paper_y = theme.hero_h + 18
            _draw_glass_panel(
                canvas,
                box=(28, paper_y, theme.width - 28, paper_y + body.height + 40),
                radius=16,
                blur_radius=10,
                overlay_fill=theme.paper_bg,
                outline=theme.paper_border,
                outline_width=1,
            )
            canvas.alpha_composite(body, (50, paper_y + 20))
            out = canvas.crop(
                (0, 0, theme.width, min(canvas.height, paper_y + body.height + 60))
            )

        out_dir = get_plugin_data_dir("render_fallback")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pillow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        out.convert("RGB").save(out_path, format="JPEG", quality=92, optimize=True)
        return out_path.resolve()
    except Exception:
        astrbot_logger.error("[dailynews] pillow fallback render failed", exc_info=True)
        return None
