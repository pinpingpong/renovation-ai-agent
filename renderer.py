"""
renderer.py — Multimodal material-swap renderer for the Renovation AI Agent.

Sends the base room photo plus one reference swatch per target surface to a
Gemini image model, with a prompt engineered to preserve geometry, perspective
and lighting while replacing only the named surfaces.

Swatches are fetched from the supplier's official image URL when available
(via an SSRF-guarded downloader and a self-cleaning temporary directory) and
otherwise synthesised from the material's colour profile.
"""

from __future__ import annotations

import io
import logging
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError

from agent_logic import AgentError, Combination, Material, Surface, friendly_api_error, safe_get

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"  # Nano Banana 2
PREFERRED_IMAGE_MODELS = (
    "gemini-3.1-flash-image",       # Nano Banana 2: best fidelity / cost balance for edits
    "gemini-3.1-flash-lite-image",  # Nano Banana 2 Lite: cheapest, lighter on quota
    "gemini-nano-banana-2.1",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image",           # highest quality, highest cost
)
MAX_BASE_SIDE = 1536
SWATCH_SIDE = 512
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SWATCH_BYTES = 8 * 1024 * 1024
SUPPORTED_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

# Neutral fallback colours when a material has no hex profile.
_CATEGORY_FALLBACK_HEX = {
    "solid": "#D9D2C5", "stone": "#D8CBB6", "plaster": "#E3D6C3", "concrete": "#B9B3AA",
    "terrazzo": "#D7CFC3", "wood": "#C9A77C", "textile": "#CFC5B5", "metallic": "#A89F91",
    "fluted": "#D4CABA", "pattern": "#C8BFB0", "other": "#CCCCCC",
}


@dataclass
class Swatch:
    material: Material
    image: Image.Image
    origin: str  # "official" or "synthesised"


@dataclass
class RenderResult:
    combination_id: str
    combination_title: str
    image_bytes: bytes
    mime_type: str
    model_note: str
    swatch_origins: List[Tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Image preparation
# --------------------------------------------------------------------------- #


def prepare_base_image(data: bytes) -> Image.Image:
    """Validate, EXIF-orient, RGB-convert and downscale the uploaded room photo."""
    if not data:
        raise AgentError("The uploaded room photo is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise AgentError("The room photo is larger than 15 MB — please upload a smaller image.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise AgentError("The uploaded file isn't a readable image (use JPG, PNG or WEBP).") from exc
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_BASE_SIDE, MAX_BASE_SIDE), Image.LANCZOS)
    return img


def closest_aspect_ratio(img: Image.Image) -> str:
    target = img.width / img.height
    def ratio(r: str) -> float:
        w, h = r.split(":")
        return int(w) / int(h)
    return min(SUPPORTED_RATIOS, key=lambda r: abs(ratio(r) - target))


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def synthesise_swatch(material: Material, side: int = SWATCH_SIDE) -> Image.Image:
    """Create a plausible, text-free swatch from the material's colour profile.

    Adds a subtle surface texture so the image model reads it as a material,
    not a flat paint chip. Deterministic per material for stable previews.
    """
    hex_color = material.hex_color or _CATEGORY_FALLBACK_HEX.get(material.category, "#CCCCCC")
    r, g, b = _hex_to_rgb(hex_color)
    rng = random.Random(f"{material.supplier}{material.code}")
    base = Image.new("RGB", (side, side), (r, g, b))

    amp = {"stone": 22, "terrazzo": 26, "plaster": 12, "concrete": 16, "textile": 10, "wood": 18}.get(
        material.category, 5
    )
    noise = Image.effect_noise((side, side), amp).convert("L")
    if material.category == "wood":
        noise = noise.resize((side, max(4, side // 32))).resize((side, side), Image.BICUBIC)
    elif material.category == "fluted":
        stripes = Image.new("L", (side, 1))
        stripes.putdata([128 + int(40 * ((x // 16) % 2 - 0.5)) for x in range(side)])
        noise = stripes.resize((side, side))
    else:
        noise = noise.filter(ImageFilter.GaussianBlur(radius=1.2 if amp > 10 else 0.6))
    # Add the zero-centred texture to the base colour (keeps the hue intact).
    offset = (np.asarray(noise, dtype=np.float32) - 128.0)[..., None] * 0.6
    textured = Image.fromarray(np.clip(np.asarray(base, dtype=np.float32) + offset, 0, 255).astype(np.uint8))

    if material.category in ("stone", "terrazzo"):
        # A few soft veins / chips for stone-like readings.
        draw = ImageDraw.Draw(textured)
        vein = tuple(max(0, c - 28) for c in (r, g, b))
        for _ in range(4 if material.category == "stone" else 60):
            x0, y0 = rng.randint(0, side), rng.randint(0, side)
            if material.category == "stone":
                pts = [(x0, y0)]
                for _ in range(8):
                    x0 += rng.randint(-40, 80)
                    y0 += rng.randint(20, 70)
                    pts.append((x0, y0))
                draw.line(pts, fill=vein, width=rng.randint(1, 3))
            else:
                rad = rng.randint(2, 7)
                shift = rng.randint(-45, 30)
                chip = tuple(min(255, max(0, c + shift)) for c in (r, g, b))
                draw.ellipse([x0, y0, x0 + rad, y0 + rad], fill=chip)
        textured = textured.filter(ImageFilter.GaussianBlur(radius=0.8))
    return textured


def _download_swatch(url: str, tmp_dir: Path) -> Optional[Image.Image]:
    """Fetch an official swatch into ``tmp_dir``; return None on any problem."""
    try:
        resp = safe_get(url, stream=True)
        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code >= 400 or not ctype.startswith("image/"):
            resp.close()
            return None
        path = tmp_dir / f"swatch_{abs(hash(url))}.img"
        written = 0
        with path.open("wb") as fh:
            for chunk in resp.iter_content(64 * 1024):
                written += len(chunk)
                if written > MAX_SWATCH_BYTES:
                    resp.close()
                    return None
                fh.write(chunk)
        with Image.open(path) as im:
            im.load()
            img = ImageOps.exif_transpose(im).convert("RGB")
        # Centre-crop to square so swatches read consistently.
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((SWATCH_SIDE, SWATCH_SIDE), Image.LANCZOS)
        return img
    except Exception as exc:  # network, decode, SSRF guard — all non-fatal
        logger.info("Swatch download failed for %s: %r", url, exc)
        return None


def load_swatches(materials: List[Material]) -> List[Swatch]:
    """Resolve a swatch per material. Temp files are removed on exit."""
    swatches: List[Swatch] = []
    with tempfile.TemporaryDirectory(prefix="renovation_swatches_") as tmp:
        tmp_dir = Path(tmp)
        for m in materials:
            img = _download_swatch(m.swatch_image_url, tmp_dir) if m.swatch_image_url else None
            if img is not None:
                swatches.append(Swatch(m, img, "official"))
            else:
                swatches.append(Swatch(m, synthesise_swatch(m), "synthesised"))
    return swatches


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

_MATERIAL_NOUN = {
    "cabinet": "laminate", "wall": "wall finish", "countertop": "countertop material",
    "backsplash": "tile", "floor": "flooring",
}
_SURFACE_GUIDANCE = {
    "cabinet": "keep door count, panel lines, handles and gaps; keep grain or veining direction consistent across doors",
    "wall": "repaint only the wall planes; keep switches, sockets, trims and anything mounted on the wall",
    "countertop": "follow the worktop's edges and thickness; keep the sink, hob and items on it",
    "backsplash": "keep the tile grid at a believable scale and keep outlets and fixtures",
    "floor": "follow the floor perspective and plank or tile direction; keep rugs and furniture legs",
}


def build_render_prompt(
    assignments: List[Tuple[Surface, Swatch]],
    room_type: str,
    extra_instructions: str = "",
) -> str:
    lines = []
    for i, (surface, sw) in enumerate(assignments, start=2):
        m = sw.material
        lines.append(
            f"- IMAGE {i} → apply to the {surface.label.upper()}: {m.supplier} {m.code} “{m.name}”, "
            f"{m.category} {_MATERIAL_NOUN.get(surface.surface_type, 'finish')}, "
            f"{m.finish or 'matte'} finish, {m.color_family or m.hex_color}. {m.description} "
            f"({_SURFACE_GUIDANCE.get(surface.surface_type, '')})"
        )

    return f"""TASK: Photorealistic material replacement (inpainting) on a real {room_type.lower()} photo.

IMAGE 1 is the BASE ROOM PHOTO. Every following image is a flat reference SWATCH of a finish.

SURFACE ASSIGNMENTS
{chr(10).join(lines)}

STRICT PRESERVATION RULES — these override everything else:
1. Keep the exact camera position, framing, lens, perspective and vanishing points of IMAGE 1.
2. Keep all geometry identical: cabinet/door count, sizes, panel lines, handles, gaps, edges,
   countertops, appliances, walls, ceiling, floor, windows and every object's position.
3. Keep the original lighting: light direction, colour temperature, shadows, reflections and exposure.
4. Change ONLY the surface finish of the assigned surfaces above. Every other surface and object
   stays pixel-faithful to IMAGE 1.
5. Map each swatch at real-world scale onto its surface: follow the surface's perspective,
   wrap correctly around edges and respect the finish's sheen.
6. Do not add, remove or restyle furniture, décor, fixtures or text. No people, no watermarks.

{('ADDITIONAL CLIENT NOTES: ' + extra_instructions.strip()) if extra_instructions.strip() else ''}
OUTPUT: one edited photograph at the same aspect ratio as IMAGE 1."""


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _extract_image(response) -> Tuple[Optional[bytes], str, str]:
    note_parts: List[str] = []
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data, inline.mime_type or "image/png", " ".join(note_parts)
            if getattr(part, "text", None):
                note_parts.append(part.text.strip())
    return None, "", " ".join(note_parts)


def _blocked_reason(response) -> Optional[str]:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return str(feedback.block_reason)
    for cand in getattr(response, "candidates", None) or []:
        reason = str(getattr(cand, "finish_reason", "") or "")
        if any(k in reason for k in ("SAFETY", "PROHIBITED", "IMAGE_SAFETY", "BLOCKLIST")):
            return reason
    return None


def render_combination(
    client: genai.Client,
    base_image: Image.Image,
    combination: Combination,
    *,
    room_type: str = "Kitchen",
    model: str = DEFAULT_IMAGE_MODEL,
    extra_instructions: str = "",
    surface_keys: Optional[List[str]] = None,
) -> RenderResult:
    """Render the scheme; ``surface_keys`` limits which surfaces change (default: all)."""
    chosen = [(sf, m) for sf, m in combination.assignments if surface_keys is None or sf.key in surface_keys]
    if not chosen:
        raise AgentError("Pick at least one surface to change before rendering.")
    swatches = load_swatches([m for _, m in chosen])
    assignments = [(sf, sw) for (sf, _), sw in zip(chosen, swatches)]
    prompt = build_render_prompt(assignments, room_type, extra_instructions)

    contents: list = [
        prompt,
        "IMAGE 1 — BASE ROOM PHOTO:",
        types.Part.from_bytes(data=_to_png_bytes(base_image), mime_type="image/png"),
    ]
    for i, (surface, sw) in enumerate(assignments, start=2):
        contents.append(f"IMAGE {i} — SWATCH for the {surface.label}:")
        contents.append(types.Part.from_bytes(data=_to_png_bytes(sw.image), mime_type="image/png"))

    def _call(with_ratio: bool):
        cfg = dict(response_modalities=["IMAGE", "TEXT"], temperature=0.4)
        if with_ratio:
            cfg["image_config"] = types.ImageConfig(aspect_ratio=closest_aspect_ratio(base_image))
        return client.models.generate_content(
            model=model, contents=contents, config=types.GenerateContentConfig(**cfg)
        )

    try:
        try:
            response = _call(with_ratio=True)
        except genai_errors.APIError as exc:
            if getattr(exc, "code", None) != 400:
                raise
            logger.warning("image_config rejected, retrying without aspect ratio: %s", exc)
            response = _call(with_ratio=False)
    except Exception as exc:
        raise friendly_api_error(exc, "rendering the combination") from exc

    blocked = _blocked_reason(response)
    data, mime, note = _extract_image(response)
    if data is None:
        if blocked:
            raise AgentError(
                "The image model declined this request on safety grounds. "
                "Try a different room photo or simplify the extra instructions.",
                detail=blocked,
            )
        raise AgentError(
            "The image model did not return an image. Please try again — "
            "image generation can occasionally return text only.",
            detail=note[:500] or None,
        )

    return RenderResult(
        combination_id=combination.id,
        combination_title=combination.title,
        image_bytes=data,
        mime_type=mime,
        model_note=note,
        swatch_origins=[(f"{sf.label}: {sw.material.label}", sw.origin) for sf, sw in assignments],
    )
