"""
showcase.py — Save a finished run as a static, key-free showcase and load it back.

A showcase is a ``showcase/`` folder committed to the repo:

    showcase/
      showcase.json   # brief, curated materials, combinations, render metadata
      before.png      # the base room photo
      render_1.png …  # the generated renders

The app displays it to every visitor without any API calls, so reviewers can see
a real end-to-end result without a key or billing.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SHOWCASE_DIR = Path(__file__).parent / "showcase"
SCHEMA_VERSION = 1


@dataclass
class ShowcaseRender:
    title: str
    image: bytes
    swatches: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class Showcase:
    meta: Dict[str, Any]
    before: Optional[bytes]
    renders: List[ShowcaseRender]
    materials: List[Dict[str, Any]]
    combinations: List[Dict[str, Any]]


def _material_dict(m) -> Dict[str, Any]:
    d = asdict(m)
    return {k: d.get(k) for k in (
        "code", "name", "supplier", "collection", "category", "finish", "color_family",
        "hex_color", "swatch_image_url", "product_url", "surface_type", "approved", "score", "reason",
    )}


def build_showcase_zip(
    *,
    brief: str,
    room_type: str,
    urls: Sequence[str],
    text_model: str,
    image_model: str,
    evaluation,
    combinations,
    renders,
    base_png: Optional[bytes],
) -> bytes:
    """Package the current session into a zip that unpacks to ``showcase/``."""
    render_entries = []
    files: Dict[str, bytes] = {}
    # Gallery lists newest first; store oldest first so numbering reads naturally.
    for i, r in enumerate(reversed(renders), start=1):
        ext = "png" if "png" in r.mime_type else "jpg"
        name = f"render_{i}.{ext}"
        files[name] = r.image_bytes
        render_entries.append({"title": r.combination_title, "file": name, "swatches": r.swatch_origins})
    if base_png:
        files["before.png"] = base_png

    payload = {
        "schema": SCHEMA_VERSION,
        "meta": {
            "created": date.today().isoformat(),
            "brief": brief.strip(),
            "room_type": room_type,
            "urls": list(urls),
            "text_model": text_model,
            "image_model": image_model,
            "style_summary": evaluation.style_summary if evaluation else "",
            "excluded_categories": evaluation.excluded_categories if evaluation else [],
            "must_haves": evaluation.must_haves if evaluation else [],
        },
        "materials": [_material_dict(m) for m in (evaluation.materials if evaluation else [])],
        "combinations": [
            {
                "title": c.title,
                "assignments": [{"surface": sf.label, "material": _material_dict(m)} for sf, m in c.assignments],
                "rationale": c.rationale,
                "principles": c.principles,
            }
            for c in combinations
        ],
        "renders": render_entries,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("showcase/showcase.json", json.dumps(payload, indent=2, ensure_ascii=False))
        for name, data in files.items():
            zf.writestr(f"showcase/{name}", data)
    return buf.getvalue()


def showcase_version(directory: Optional[Path] = None) -> float:
    """Modification time of the manifest (0 when absent) — used as a cache key."""
    manifest = (directory or SHOWCASE_DIR) / "showcase.json"
    return manifest.stat().st_mtime if manifest.is_file() else 0.0


def load_showcase(directory: Optional[Path] = None) -> Optional[Showcase]:
    """Load ``showcase/`` if present and valid; return None otherwise (never raises)."""
    directory = directory or SHOWCASE_DIR
    manifest = directory / "showcase.json"
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        before_path = directory / "before.png"
        renders = []
        for entry in payload.get("renders", []):
            path = directory / Path(entry["file"]).name  # no path traversal
            if path.is_file():
                renders.append(ShowcaseRender(
                    title=entry.get("title", ""),
                    image=path.read_bytes(),
                    swatches=[tuple(s) for s in entry.get("swatches", [])],
                ))
        return Showcase(
            meta=payload.get("meta", {}),
            before=before_path.read_bytes() if before_path.is_file() else None,
            renders=renders,
            materials=payload.get("materials", []),
            combinations=payload.get("combinations", []),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
