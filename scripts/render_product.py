"""Draw the official VocalLift mark and product card.

The booth serves the files this writes:

    web/mark.png    app icon and browser tab
    web/og.png      the share image

Regenerate them with Pillow installed:

    pip install pillow
    python scripts/render_product.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

INK = np.array([20, 17, 14], dtype=np.float32)
COPPER = (227, 154, 86)
COPPER_DEEP = (196, 98, 45)
CREAM = (244, 238, 230)
MUTED = (179, 167, 154)

# The header mark, in a 32px viewBox: three curves across the middle.
_WAVE = [
    ((6, 18), (8, 12), (10, 12), (12, 18)),
    ((12, 18), (14, 24), (16, 24), (18, 18)),
    ((18, 18), (20, 12), (22, 12), (24, 18)),
]


def _font(size: int, serif: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [r"C:\Windows\Fonts\georgia.ttf", r"C:\Windows\Fonts\pala.ttf", r"C:\Windows\Fonts\times.ttf"]
        if serif
        else [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _curve(samples: int = 48) -> np.ndarray:
    points = []
    for index, (p0, p1, p2, p3) in enumerate(_WAVE):
        steps = np.linspace(0, 1, samples, endpoint=index == len(_WAVE) - 1)
        t = steps[:, None]
        a, b, c, d = (np.array(p, dtype=np.float32) for p in (p0, p1, p2, p3))
        points.append((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t**2 * c + t**3 * d)
    return np.vstack(points)


def _background(width: int, height: int, glow: tuple[float, float], radius: float) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    image = np.broadcast_to(INK, (height, width, 3)).copy()
    gx, gy = glow
    falloff = np.exp(-(((x - gx) ** 2 + (y - gy) ** 2) / (2 * radius**2)))
    image += falloff[..., None] * (np.array(COPPER_DEEP, dtype=np.float32) - INK) * 0.55
    vignette = np.sqrt(((x - width / 2) / (width * 0.72)) ** 2 + ((y - height / 2) / (height * 0.78)) ** 2)
    image *= (1 - np.clip(vignette, 0, 1) ** 1.6 * 0.42)[..., None]
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")


def _draw_mark(base: Image.Image, center: tuple[float, float], radius: float) -> None:
    """Copper ring and the cream waveform, with a soft halo under the line."""
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = center
    ring = radius * 0.9
    box = [cx - ring, cy - ring, cx + ring, cy + ring]
    width = max(2, int(round(radius * 0.045)))
    draw.ellipse(box, outline=COPPER + (255,), width=width)

    wave = _curve()
    scale = radius * 0.046
    points = [(cx + (x - 16) * scale, cy + (y - 16) * scale) for x, y in wave]
    stroke = max(3, int(round(radius * 0.055)))
    halo = Image.new("RGBA", base.size, (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo)
    halo_draw.line(points, fill=COPPER + (170,), width=stroke * 3, joint="curve")
    _caps(halo_draw, points, stroke * 3, COPPER + (170,))
    overlay = Image.alpha_composite(overlay, halo.filter(ImageFilter.GaussianBlur(radius * 0.06)))
    cream = ImageDraw.Draw(overlay)
    cream.line(points, fill=CREAM + (255,), width=stroke, joint="curve")
    _caps(cream, points, stroke, CREAM + (255,))
    base.paste(overlay, mask=overlay.getchannel("A"))


def _caps(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], stroke: float, fill: tuple[int, ...]) -> None:
    radius = stroke / 2
    for px, py in (points[0], points[-1]):
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill)


def _tracked(draw: ImageDraw.ImageDraw, origin: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], tracking: float) -> None:
    x, y = origin
    for character in text:
        draw.text((x, y), character, font=font, fill=fill)
        x += draw.textlength(character, font=font) + tracking


def render_mark(size: int = 1024) -> Image.Image:
    image = _background(size, size, (size * 0.5, size * 0.42), size * 0.34)
    _draw_mark(image, (size / 2, size / 2), size * 0.36)
    return image


def render_card(width: int = 1920, height: int = 1080) -> Image.Image:
    image = _background(width, height, (width * 0.28, height * 0.48), height * 0.42)
    _draw_mark(image, (width * 0.27, height * 0.5), height * 0.3)

    draw = ImageDraw.Draw(image)
    x = int(width * 0.48)
    kicker = _font(int(height * 0.028), serif=False)
    name = _font(int(height * 0.12), serif=True)
    line = _font(int(height * 0.046), serif=True)
    _tracked(draw, (x, int(height * 0.3)), "SINGING ENGINE", kicker, COPPER, height * 0.006)
    draw.text((x, int(height * 0.36)), "VocalLift", font=name, fill=CREAM)
    rule_y = int(height * 0.54)
    draw.line([(x, rule_y), (x + int(width * 0.12), rule_y)], fill=COPPER, width=max(2, height // 360))
    draw.text((x, int(height * 0.58)), "Land the note.", font=line, fill=CREAM)
    draw.text((x, int(height * 0.66)), "Keep the voice.", font=line, fill=MUTED)
    return image


def main() -> None:
    WEB.mkdir(parents=True, exist_ok=True)
    mark = WEB / "mark.png"
    card = WEB / "og.png"
    render_mark().save(mark, "PNG")
    render_card().save(card, "PNG")
    print(f"wrote {mark.relative_to(ROOT)}")
    print(f"wrote {card.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
