"""
agent_logic.py — Search-grounded catalog extractor, constraint validator and
combination engine for the Renovation AI Agent.

Pipeline
--------
1. ``parse_urls``            – normalise / validate the user's supplier links.
2. ``resolve_catalog``       – Gemini + URL Context + Google Search grounding
                               resolves each link into real product codes.
3. ``evaluate_materials``    – Gemini (structured output) scores every item
                               against the user's free-text constraints; a
                               deterministic validator then enforces any hard
                               category exclusions the model identified.
4. ``build_combinations``    – Gemini proposes structural pairings
                               (base / feature / accent); results are validated
                               and backed by a deterministic heuristic fallback.

Every public function raises ``AgentError`` with a user-safe message instead of
leaking raw tracebacks to the UI.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_TEXT_MODEL = "gemini-flash-latest"  # alias that always points at the current Flash
# Preferred text models, best first; the first one the key can access is the default.
# 3.5 Flash-Lite leads: verified end to end (grounding, evaluation, pairing). Gemini 2.5 models can
# appear in models.list() yet return 404 for newer projects, so they come last.
PREFERRED_TEXT_MODELS = (
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.8-flash",
    "gemini-3.6-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-2.5-flash",
)
# Substrings marking models that can't serve this app's text or image calls.
_NON_GENERATION_MARKERS = (
    "tts", "live", "embedding", "transcribe", "audio", "robotics", "lyria",
    "research", "antigravity", "imagen", "veo", "aqa", "translate",
)
MAX_URLS = 6
MAX_TOTAL_ITEMS = 150
HTTP_TIMEOUT = 10  # seconds
API_TIMEOUT_MS = 180_000
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 RenovationAIAgent/1.0"
)

CATEGORIES = (
    "solid",      # plain colours
    "stone",      # marble, limestone, travertine, granite, sandstone
    "plaster",    # plaster / limewash / microcement looks
    "concrete",   # cement, concrete, terrazzo-free industrial
    "terrazzo",
    "wood",       # any timber / wood-grain look
    "textile",    # fabric, linen, leather, rattan weaves
    "metallic",   # brushed metal, bronze, aluminium
    "fluted",     # 3D / vertical fluting / ribbed panels
    "pattern",    # graphic / abstract patterns
    "other",
)

# What a supplier link can cover. Each type maps to one or more surfaces in the room.
SURFACE_TYPES = {
    "cabinet": "Cabinet / carpentry laminate",
    "wall": "Wall paint / wall finish",
    "countertop": "Countertop / vanity top",
    "backsplash": "Backsplash / wall tiles",
    "floor": "Flooring",
}
# Room-specific names for each surface. Cabinets get base / feature / optional accent.
ROOM_SURFACE_LABELS: Dict[str, Dict[str, Any]] = {
    "Kitchen": {"cabinet": ("Lower cabinets", "Upper cabinets", "Tall unit / accent panel"),
                "wall": "Walls", "countertop": "Countertop", "backsplash": "Backsplash", "floor": "Floor"},
    "Bedroom": {"cabinet": ("Wardrobe carcass & frame", "Wardrobe doors", "Headboard / feature panel"),
                "wall": "Walls", "countertop": "Dressing table top", "backsplash": "Feature wall tiles",
                "floor": "Floor"},
    "Living room": {"cabinet": ("TV console & low cabinets", "Feature wall panel", "Display niche accent"),
                    "wall": "Walls", "countertop": "Console top", "backsplash": "Feature wall tiles",
                    "floor": "Floor"},
    "Bathroom": {"cabinet": ("Vanity cabinet", "Mirror cabinet", "Niche accent"),
                 "wall": "Walls", "countertop": "Vanity top", "backsplash": "Wall tiles", "floor": "Floor tiles"},
    "Study": {"cabinet": ("Desk & base cabinets", "Overhead / bookshelf panels", "Accent strip"),
              "wall": "Walls", "countertop": "Desk top", "backsplash": "Feature wall", "floor": "Floor"},
}
# Image models lose precision with many swatches, so a scheme covers at most this many surfaces.
MAX_SCHEME_SURFACES = 5

# Categories that make good grounding "base" surfaces vs expressive "feature"
# surfaces. Used only by the deterministic fallback pairing engine.
BASE_CATEGORIES = {"solid", "plaster", "concrete"}
FEATURE_CATEGORIES = {"stone", "terrazzo", "wood", "textile", "fluted", "pattern", "metallic"}


# --------------------------------------------------------------------------- #
# Errors & client
# --------------------------------------------------------------------------- #


class AgentError(Exception):
    """An error whose ``message`` is safe and helpful to show to end users."""

    def __init__(self, message: str, *, detail: Optional[str] = None, kind: str = "other"):
        super().__init__(message)
        self.message = message
        self.detail = detail
        # "quota", "auth", "model" or "other". Account-wide kinds stop the pipeline early.
        self.kind = kind


# Errors that will fail every later request too, so retrying other URLs is pointless.
FATAL_KINDS = ("quota", "auth", "model")


def make_client(api_key: str) -> genai.Client:
    if not api_key or not api_key.strip():
        raise AgentError("Add your Gemini API key in the sidebar to run the agent.")
    try:
        return genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(
                timeout=API_TIMEOUT_MS,
                # Absorb short per-minute rate limits and transient overloads.
                retry_options=types.HttpRetryOptions(
                    attempts=3, initial_delay=4.0, max_delay=20.0, http_status_codes=[429, 503]
                ),
            ),
        )
    except Exception as exc:  # pragma: no cover - SDK construction rarely fails
        raise AgentError("Could not initialise the Gemini client.", detail=repr(exc)) from exc


def list_generation_models(client: genai.Client) -> Tuple[List[str], List[str]]:
    """Return (text_models, image_models) this key can call with generateContent."""
    text: set = set()
    image: set = set()
    try:
        for m in client.models.list():
            name = (m.name or "").split("/")[-1]
            actions = m.supported_actions or []
            if not name.startswith("gemini") or (actions and "generateContent" not in actions):
                continue
            if any(marker in name for marker in _NON_GENERATION_MARKERS):
                continue
            (image if ("image" in name or "nano-banana" in name) else text).add(name)
    except Exception as exc:
        raise friendly_api_error(exc, "listing the models available to your key") from exc
    return sorted(text, reverse=True), sorted(image, reverse=True)


def check_model_access(
    client: genai.Client, model: str, image_model: Optional[str] = None
) -> List[Tuple[str, bool, str]]:
    """Tiny probes: a plain request, one with Google Search grounding, and
    (optionally) one small image generation.

    Returns [(check_name, ok, message)] so the UI can show whether quota problems
    come from the model itself or from the grounding tool.
    """
    probes = [
        ("Plain request", None),
        ("Google Search grounding", [types.Tool(google_search=types.GoogleSearch())]),
    ]
    results: List[Tuple[str, bool, str]] = []
    for label, tools in probes:
        try:
            resp = client.models.generate_content(
                model=model,
                contents="Reply with the single word OK.",
                config=types.GenerateContentConfig(tools=tools, temperature=0),
            )
            results.append((label, True, f"{model} replied: {(resp.text or '').strip()[:40] or '(empty)'}"))
        except Exception as exc:
            err = friendly_api_error(exc, f"testing {model}")
            results.append((label, False, err.detail or err.message))
            if err.kind == "auth" or label == "Plain request":
                return results  # nothing else can work if a plain request doesn't
            break

    if image_model:
        label = f"Image generation ({image_model})"
        try:
            resp = client.models.generate_content(
                model=image_model,
                contents="Generate a small plain swatch of warm beige plaster texture.",
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
            got_image = any(
                getattr(getattr(p, "inline_data", None), "data", None)
                for c in (resp.candidates or [])
                for p in (getattr(c.content, "parts", None) or [])
            )
            results.append((label, got_image, "Returned an image." if got_image else "Replied without an image."))
        except Exception as exc:
            err = friendly_api_error(exc, f"testing {image_model}")
            results.append((label, False, err.detail or err.message))
    return results


# Stable model families, matched exactly so preview / exp / dated variants are never picked.
_STABLE_FAMILIES = {
    "text": re.compile(r"^gemini-(\d+(?:\.\d+)*)-flash-lite$"),
    "image": re.compile(r"^gemini-(\d+(?:\.\d+)*)-flash-image$"),
}


def newest_stable_model(available: Iterable[str], family: str) -> Optional[str]:
    """Highest-versioned stable model of a family, e.g. gemini-4-flash-lite over gemini-3.5-flash-lite."""
    pattern = _STABLE_FAMILIES[family]
    best: Optional[Tuple[Tuple[int, ...], str]] = None
    for name in available:
        m = pattern.match(name)
        if m:
            version = tuple(int(part) for part in m.group(1).split("."))
            if best is None or version > best[0]:
                best = (version, name)
    return best[1] if best else None


def pick_default_model(
    available: List[str], preferred: Iterable[str], family: Optional[str] = None
) -> Optional[str]:
    """Newest stable model of ``family`` if one is available, else the first preferred model,
    else the first non-lite flash, else the first model listed."""
    if family:
        newest = newest_stable_model(available, family)
        if newest:
            return newest
    for name in preferred:
        if name in available:
            return name
    flash = [n for n in available if "flash" in n and "lite" not in n]
    return (flash or available or [None])[0]


def friendly_api_error(exc: Exception, action: str) -> AgentError:
    """Translate SDK / network exceptions into a user-facing ``AgentError``."""
    if isinstance(exc, AgentError):
        return exc

    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        raw = (getattr(exc, "message", None) or str(exc))[:500]
        full = str(exc)[:2000]
        lowered = full.lower()
        if code == 429 or "resource_exhausted" in lowered or "quota" in lowered:
            model = re.search(r"model:\s*([\w\-]+(?:\.[\w\-]+)*)", full)
            model_name = model.group(1) if model else "the selected model"
            retry = re.search(r"retry in ([\d.]+)s", full, re.I)
            has_metric = "quota_metric" in lowered or "quotafailure" in lowered or "limit:" in lowered
            if not has_metric and not retry:
                # Generic RESOURCE_EXHAUSTED with no per-model metric: account / project level.
                msg = (
                    f"Google refused the request while {action} with a generic ‘exceeded your current "
                    "quota’ error. It carries no per-minute or per-model detail, so it comes from the "
                    "API key's Google Cloud project, not the model choice: usually billing isn't set "
                    "up, the free tier isn't available to the project, or prepaid credit has run "
                    "out. Check usage at https://ai.dev/rate-limit and billing in Google AI Studio, "
                    "or use the ‘Test key & models’ button in the sidebar."
                )
            elif re.search(r"limit:\s*0\b", full):
                msg = (
                    f"Your API key has no quota for {model_name} on its current plan, so the request "
                    f"was refused while {action}. Pick another model under ‘Advanced model settings’ "
                    "(a ‘flash-lite’ model usually has free-tier quota), or enable billing for your "
                    "project in Google AI Studio."
                )
            elif "per_day" in lowered or "perday" in lowered or "daily" in lowered:
                msg = (
                    f"Your daily Gemini quota for {model_name} is used up ({action}). "
                    "Try again tomorrow, switch models in ‘Advanced model settings’, or enable billing."
                )
            else:
                wait = f"about {round(float(retry.group(1)))} seconds" if retry else "a minute"
                msg = (
                    f"Gemini's per-minute rate limit for {model_name} was hit while {action}, even after "
                    f"automatic retries. Wait {wait} and try again, or lower ‘Max products per URL’."
                )
            return AgentError(msg, detail=f"HTTP {code}: {full}", kind="quota")
        elif code in (401, 403) or "api key" in lowered or "api_key" in lowered:
            msg = "Gemini rejected the API key. Check that it is correct and has the Gemini API enabled."
            return AgentError(msg, detail=f"HTTP {code}: {raw}", kind="auth")
        elif code == 404 or ("not found" in lowered and "model" in lowered):
            msg = (
                f"The selected Gemini model is not available for this key while {action}. "
                "Pick a different model under ‘Advanced model settings’ in the sidebar."
            )
            return AgentError(msg, detail=f"HTTP {code}: {raw}", kind="model")
        elif code == 400:
            msg = f"Gemini could not process the request while {action}: {raw}"
        elif code and code >= 500:
            msg = f"Gemini is temporarily unavailable ({code}) while {action}. Please retry shortly."
        else:
            msg = f"Gemini returned an error while {action}."
        return AgentError(msg, detail=f"HTTP {code}: {raw}")

    if isinstance(exc, requests.Timeout):
        return AgentError(f"A network request timed out while {action}.", detail=repr(exc))
    if isinstance(exc, requests.RequestException):
        return AgentError(f"A network error occurred while {action}.", detail=repr(exc))

    logger.exception("Unexpected error while %s", action)
    return AgentError(f"Something unexpected went wrong while {action}.", detail=repr(exc))


# --------------------------------------------------------------------------- #
# Safe HTTP (SSRF-guarded) — used for URL probing and swatch downloads
# --------------------------------------------------------------------------- #


def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def is_public_http_url(url: str) -> bool:
    """True when ``url`` is http(s) and resolves only to public addresses."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    return _host_is_public(parsed.hostname)


def safe_get(url: str, *, stream: bool = False, max_redirects: int = 4) -> requests.Response:
    """GET that re-validates every redirect hop against private networks."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    current = url
    for _ in range(max_redirects + 1):
        if not is_public_http_url(current):
            raise AgentError(f"Blocked a request to a non-public or invalid address: {current}")
        resp = session.get(current, timeout=HTTP_TIMEOUT, stream=stream, allow_redirects=False)
        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                break
            current = urljoin(current, location)
            continue
        return resp
    raise AgentError(f"Too many redirects while fetching {url}")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Material:
    id: str
    code: str
    name: str
    supplier: str = ""
    collection: str = ""
    category: str = "other"
    finish: str = ""
    color_family: str = ""
    hex_color: str = ""
    description: str = ""
    swatch_image_url: str = ""
    product_url: str = ""
    source_url: str = ""
    surface_type: str = "cabinet"  # key of SURFACE_TYPES
    # Populated by the constraint evaluator
    approved: Optional[bool] = None
    score: int = 0
    reason: str = ""
    attributes: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        bits = [self.code, self.name]
        if self.supplier:
            bits.append(f"({self.supplier})")
        return " ".join(b for b in bits if b)

    def prompt_line(self) -> str:
        return (
            f"{self.id} | type={self.surface_type} | {self.supplier} {self.code} “{self.name}” | category={self.category}"
            f" | finish={self.finish or 'n/a'} | colour={self.color_family or 'n/a'} {self.hex_color}"
            f" | {self.description}"
        )


@dataclass
class UrlReport:
    url: str
    reachable: Optional[bool]
    page_title: str = ""
    items_found: int = 0
    supplier: str = ""
    collection: str = ""
    note: str = ""
    surface_type: str = ""
    detected: bool = True  # False when the user chose the surface type


@dataclass
class CatalogResult:
    materials: List[Material]
    reports: List[UrlReport]
    sources: List[Dict[str, str]]
    warnings: List[str]


@dataclass
class Evaluation:
    style_summary: str
    excluded_categories: List[str]
    must_haves: List[str]
    materials: List[Material]

    @property
    def approved(self) -> List[Material]:
        return [m for m in self.materials if m.approved]


@dataclass(frozen=True)
class Surface:
    key: str            # unique id, e.g. "cabinet_base", "wall"
    label: str          # room-specific name, e.g. "Lower cabinets"
    surface_type: str   # key of SURFACE_TYPES
    role: str = ""      # "base" / "feature" / "accent" for cabinets
    optional: bool = False


def room_surfaces(
    room_type: str,
    surface_types: Iterable[str],
    labels: Optional[Dict[str, Any]] = None,
    cabinet_materials: int = 2,
) -> List[Surface]:
    """Surfaces to design for the surface types present, in a stable order, capped."""
    names = {**ROOM_SURFACE_LABELS.get(room_type, ROOM_SURFACE_LABELS["Kitchen"]), **(labels or {})}
    present = set(surface_types)
    required: List[Surface] = []
    optional: List[Surface] = []
    if "cabinet" in present:
        base, feature, accent = names["cabinet"]
        required.append(Surface("cabinet_base", base, "cabinet", "base"))
        if cabinet_materials >= 2:
            required.append(Surface("cabinet_feature", feature, "cabinet", "feature"))
        if cabinet_materials >= 3:
            optional.append(Surface("cabinet_accent", accent, "cabinet", "accent", optional=True))
    for t in ("countertop", "wall", "backsplash", "floor"):
        if t in present:
            required.append(Surface(t, names[t], t))
    return (required + optional)[:MAX_SCHEME_SURFACES]


@dataclass
class Combination:
    id: str
    title: str
    assignments: List[Tuple[Surface, Material]]
    rationale: str
    principles: List[str]
    source: str = "agent"  # "agent" or "heuristic"

    @property
    def materials(self) -> List[Material]:
        return [m for _, m in self.assignments]


# --------------------------------------------------------------------------- #
# Step 1 — URL parsing & probing
# --------------------------------------------------------------------------- #

_URL_RE = re.compile(r"https?://[^\s,;<>\"']+", re.IGNORECASE)


def parse_urls(raw: str) -> Tuple[List[str], List[str]]:
    """Extract unique http(s) URLs from free text. Returns (urls, problems)."""
    problems: List[str] = []
    if not raw or not raw.strip():
        return [], ["Paste at least one supplier portfolio or catalogue URL."]

    found: List[str] = []
    for token in re.split(r"[\s,;]+", raw.strip()):
        if not token:
            continue
        if not token.lower().startswith(("http://", "https://")):
            if "." in token and " " not in token:
                token = "https://" + token
            else:
                problems.append(f"Ignored “{token}” — not a URL.")
                continue
        match = _URL_RE.match(token)
        if not match:
            problems.append(f"Ignored “{token}” — not a valid URL.")
            continue
        url = match.group(0).rstrip(").,")
        parsed = urlparse(url)
        if not parsed.hostname or "." not in parsed.hostname:
            problems.append(f"Ignored “{token}” — missing a valid domain.")
            continue
        if url not in found:
            found.append(url)

    if len(found) > MAX_URLS:
        problems.append(f"Only the first {MAX_URLS} URLs will be processed.")
        found = found[:MAX_URLS]
    return found, problems


def probe_url(url: str) -> UrlReport:
    """Cheap reachability check. Failure is non-fatal: grounding may still work."""
    try:
        resp = safe_get(url)
        title = ""
        if "html" in resp.headers.get("Content-Type", ""):
            m = re.search(r"<title[^>]*>(.*?)</title>", resp.text[:200_000], re.I | re.S)
            if m:
                title = re.sub(r"\s+", " ", m.group(1)).strip()[:160]
        ok = resp.status_code < 400
        note = "" if ok else f"HTTP {resp.status_code} — will rely on search grounding."
        return UrlReport(url=url, reachable=ok, page_title=title, note=note)
    except AgentError as exc:
        return UrlReport(url=url, reachable=False, note=exc.message)
    except requests.RequestException as exc:
        return UrlReport(
            url=url,
            reachable=False,
            note=f"Not reachable from the server ({type(exc).__name__}) — will rely on search grounding.",
        )


# --------------------------------------------------------------------------- #
# Step 2 — Search-grounded catalog resolution
# --------------------------------------------------------------------------- #

_RESOLVE_SYSTEM = """You are a meticulous materials librarian for interior surface finishes: \
cabinet laminates and panels (e.g. Lamitak, Jennings, EDL, Admira), wall paints, countertops, \
tiles and flooring.
You only report REAL products that are published by the supplier. Never invent product codes.
If you are unsure a code is genuine, omit that item."""

_RESOLVE_PROMPT = """Resolve the supplier portfolio / catalogue at this URL:
{url}
{title_hint}
{type_hint}
Instructions:
1. Read the page with the URL context tool. If it is blocked, JavaScript-rendered or only partially
   readable, use Google Search to find the official product listings for this same collection
   (prefer the supplier's own domain).
2. Collect up to {max_items} distinct products from this collection. Prefer variety across colours
   and patterns over near-duplicates.
3. For each product capture its official code and name plus your best visual profile.

Respond with ONLY a JSON object inside a ```json fenced block, in exactly this shape:
{{
  "supplier": "Supplier brand name",
  "collection": "Collection / range name",
  "surface_type": "what this catalogue is for — one of: {surface_types}",
  "items": [
    {{
      "code": "official product code, e.g. 'PG 5502' or 'NT 1001'",
      "name": "official product name",
      "category": "one of: {categories}",
      "finish": "e.g. matte, suede, textured, high gloss, woodgrain emboss",
      "color_family": "short plain-English colour family, e.g. 'warm sand beige'",
      "hex_color": "approximate dominant colour as #RRGGBB",
      "description": "one sentence on pattern, veining, grain or texture",
      "swatch_image_url": "direct URL to the official swatch image if you saw one, else empty string",
      "product_url": "URL of the product page if known, else empty string"
    }}
  ]
}}
"""


class _CatalogItem(BaseModel):
    code: str = ""
    name: str = ""
    category: str = "other"
    finish: str = ""
    color_family: str = ""
    hex_color: str = ""
    description: str = ""
    swatch_image_url: str = ""
    product_url: str = ""


class _CatalogPayload(BaseModel):
    supplier: str = ""
    collection: str = ""
    surface_type: str = ""
    items: List[_CatalogItem] = []


_SURFACE_ALIASES = (
    ("countertop", ("countertop", "worktop", "quartz", "solid surface", "vanity top", "benchtop", "sintered")),
    ("backsplash", ("backsplash", "splashback", "tile", "mosaic")),
    ("floor", ("floor", "vinyl", "spc", "parquet", "plank")),
    ("wall", ("wall", "paint", "limewash", "wallpaper", "coating")),
    ("cabinet", ("cabinet", "laminate", "carpentry", "panel", "veneer", "compact", "melamine")),
)


def clean_surface_type(value: str) -> str:
    """Normalise a model- or user-supplied surface type; default to cabinet laminates."""
    value = (value or "").strip().lower()
    if value in SURFACE_TYPES:
        return value
    for key, words in _SURFACE_ALIASES:
        if any(w in value for w in words):
            return key
    return "cabinet"


def _extract_json(text: str):
    """Pull the first JSON object/array out of a model response."""
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found")
    start = min(starts)
    end = max(candidate.rfind("}"), candidate.rfind("]"))
    return json.loads(candidate[start : end + 1])


_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _clean_hex(value: str) -> str:
    m = _HEX_RE.match((value or "").strip())
    return f"#{m.group(1).upper()}" if m else ""


def _clean_url(value: str) -> str:
    value = (value or "").strip()
    return value if value.lower().startswith(("http://", "https://")) else ""


def _clean_category(value: str) -> str:
    value = (value or "").strip().lower()
    if value in CATEGORIES:
        return value
    aliases = {
        "marble": "stone", "granite": "stone", "limestone": "stone", "travertine": "stone",
        "woodgrain": "wood", "timber": "wood", "oak": "wood", "walnut": "wood",
        "fabric": "textile", "linen": "textile", "leather": "textile",
        "metal": "metallic", "cement": "concrete", "plain": "solid", "unicolour": "solid",
        "3d": "fluted", "ribbed": "fluted",
    }
    for key, cat in aliases.items():
        if key in value:
            return cat
    return "other"


def _grounding_sources(response) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for cand in getattr(response, "candidates", None) or []:
        gm = getattr(cand, "grounding_metadata", None)
        for chunk in (getattr(gm, "grounding_chunks", None) or []):
            web = getattr(chunk, "web", None)
            if web is not None and getattr(web, "uri", None):
                sources.append({"title": web.title or web.uri, "uri": web.uri, "via": "Google Search"})
        um = getattr(cand, "url_context_metadata", None)
        for meta in (getattr(um, "url_metadata", None) or []):
            uri = getattr(meta, "retrieved_url", None)
            if uri:
                status = str(getattr(meta, "url_retrieval_status", "") or "").replace("URL_RETRIEVAL_STATUS_", "")
                sources.append({"title": f"URL context ({status.lower() or 'fetched'})", "uri": uri, "via": "URL Context"})
    return sources


def _grounded_call(client: genai.Client, model: str, prompt: str):
    """Run a grounded call with URL Context + Search, degrading to Search only."""
    tool_sets = [
        [types.Tool(url_context=types.UrlContext()), types.Tool(google_search=types.GoogleSearch())],
        [types.Tool(google_search=types.GoogleSearch())],
    ]
    last_exc: Optional[Exception] = None
    for tools in tool_sets:
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_RESOLVE_SYSTEM,
                    tools=tools,
                    temperature=0.2,
                ),
            )
        except genai_errors.APIError as exc:
            # Some models / regions don't support url_context — retry with search only.
            if getattr(exc, "code", None) == 400 and len(tools) > 1:
                logger.warning("URL context unavailable, retrying with search only: %s", exc)
                last_exc = exc
                continue
            raise
    raise last_exc  # type: ignore[misc]


def resolve_catalog_url(
    client: genai.Client,
    url: str,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    max_items: int = 25,
    page_title: str = "",
    surface_type: Optional[str] = None,
) -> Tuple[_CatalogPayload, List[Dict[str, str]]]:
    """Resolve one portfolio URL into structured catalogue items (grounded)."""
    title_hint = f"(The page title reported by the server is: “{page_title}”.)" if page_title else ""
    type_hint = (
        f"(The user says this catalogue is for: {SURFACE_TYPES[surface_type]}. Use surface_type "
        f"\"{surface_type}\".)" if surface_type in SURFACE_TYPES else ""
    )
    prompt = _RESOLVE_PROMPT.format(
        url=url, title_hint=title_hint, type_hint=type_hint, max_items=max_items,
        categories=", ".join(CATEGORIES), surface_types=", ".join(SURFACE_TYPES),
    )
    try:
        response = _grounded_call(client, model, prompt)
    except Exception as exc:
        raise friendly_api_error(exc, f"resolving {url}") from exc

    text = getattr(response, "text", None) or ""
    try:
        data = _extract_json(text)
        if isinstance(data, list):
            data = {"items": data}
        payload = _CatalogPayload.model_validate(data)
    except (ValueError, ValidationError) as exc:
        raise AgentError(
            f"The agent could not extract a product list from {url}. "
            "Try a more specific collection page (e.g. a ‘Solids’ or ‘Stone’ listing).",
            detail=f"{exc!r} | response head: {text[:300]!r}",
        ) from exc
    return payload, _grounding_sources(response)


def resolve_catalog(
    client: genai.Client,
    urls: Iterable[str],
    *,
    model: str = DEFAULT_TEXT_MODEL,
    max_items_per_url: int = 25,
    surface_types: Optional[Dict[str, str]] = None,
    progress=None,
) -> CatalogResult:
    """Resolve every URL; individual URL failures become warnings, not crashes.

    ``surface_types`` maps a URL to a SURFACE_TYPES key the user chose; URLs not in it
    (or mapped to "auto") are auto-detected by the model.
    """
    urls = list(urls)
    surface_types = surface_types or {}
    materials: List[Material] = []
    reports: List[UrlReport] = []
    sources: List[Dict[str, str]] = []
    warnings: List[str] = []
    seen: set = set()

    for idx, url in enumerate(urls, start=1):
        if progress:
            progress(idx, len(urls), url)
        report = probe_url(url)
        forced = surface_types.get(url)
        forced = forced if forced in SURFACE_TYPES else None
        try:
            payload, src = resolve_catalog_url(
                client, url, model=model, max_items=max_items_per_url, page_title=report.page_title,
                surface_type=forced,
            )
        except AgentError as exc:
            # Auth / quota problems will fail every URL — surface immediately.
            if exc.kind in FATAL_KINDS:
                raise
            report.note = exc.message
            warnings.append(exc.message)
            reports.append(report)
            continue

        report.supplier, report.collection = payload.supplier, payload.collection
        report.surface_type = forced or clean_surface_type(payload.surface_type)
        report.detected = forced is None
        sources.extend(src)
        for item in payload.items:
            code = (item.code or "").strip()
            name = (item.name or "").strip()
            if not code and not name:
                continue
            key = (payload.supplier.lower(), (code or name).upper().replace(" ", ""))
            if key in seen:
                continue
            seen.add(key)
            materials.append(
                Material(
                    id=f"M{len(materials) + 1:03d}",
                    code=code or "—",
                    name=name or code,
                    supplier=payload.supplier.strip(),
                    collection=payload.collection.strip(),
                    category=_clean_category(item.category),
                    finish=item.finish.strip(),
                    color_family=item.color_family.strip(),
                    hex_color=_clean_hex(item.hex_color),
                    description=item.description.strip(),
                    swatch_image_url=_clean_url(item.swatch_image_url),
                    product_url=_clean_url(item.product_url),
                    source_url=url,
                    surface_type=report.surface_type,
                )
            )
            report.items_found += 1
            if len(materials) >= MAX_TOTAL_ITEMS:
                break
        if report.items_found == 0:
            msg = f"No products could be resolved from {url}."
            report.note = report.note or msg
            warnings.append(msg)
        reports.append(report)
        if len(materials) >= MAX_TOTAL_ITEMS:
            warnings.append(f"Catalogue capped at {MAX_TOTAL_ITEMS} items.")
            break

    # De-duplicate sources while keeping order.
    uniq: Dict[str, Dict[str, str]] = {}
    for s in sources:
        uniq.setdefault(s["uri"], s)

    if not materials:
        raise AgentError(
            "No materials could be resolved from the provided URLs. "
            "Check the links point at a public product/collection page and try again.",
            detail=" | ".join(warnings) or None,
        )
    return CatalogResult(materials=materials, reports=reports, sources=list(uniq.values()), warnings=warnings)


# --------------------------------------------------------------------------- #
# Step 3 — Constraint evaluation (structured output + deterministic validator)
# --------------------------------------------------------------------------- #

_EVAL_SYSTEM = """You are a senior interior designer acting as a strict compliance reviewer.
You evaluate interior finish catalogue items (laminates, paints, countertops, tiles, flooring)
against a client's written design brief.
Hard negative rules ("no", "zero", "avoid completely", "must not") are absolute: any item that
looks like the forbidden material — including printed / faux looks — must be rejected.
Positive rules ("must include", "favour") raise scores; they do not require every item to match."""

_EVAL_PROMPT = """CLIENT DESIGN BRIEF
-------------------
{constraints}

ROOM: {room_type}

CATALOGUE ITEMS (id | surface type | supplier code “name” | category | finish | colour | description)
-------------------------------------------------------------------------------------
{items}

TASKS
1. Summarise the brief in one sentence (style_summary).
2. excluded_categories: list every category from [{categories}] that the brief forbids outright
   (e.g. "zero wood/timber" → ["wood"]). Empty list if none.
3. must_haves: short phrases for any positive requirements (e.g. "light oak woodgrain", "vertical fluting").
4. For EVERY item id above return a verdict: approved (bool), score 0-100 for brief fit,
   a concise reason (max 25 words) that cites the specific rule, and detected_attributes
   (e.g. ["warm", "matte", "stone-look"]).
"""


class _Verdict(BaseModel):
    id: str
    approved: bool
    score: int
    reason: str
    detected_attributes: List[str]


class _EvaluationPayload(BaseModel):
    style_summary: str
    excluded_categories: List[str]
    must_haves: List[str]
    verdicts: List[_Verdict]


def _structured_call(client: genai.Client, model: str, system: str, prompt: str, schema, temperature: float):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        ),
    )
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    return schema.model_validate(_extract_json(response.text or ""))


def evaluate_materials(
    client: genai.Client,
    materials: List[Material],
    constraints: str,
    *,
    room_type: str = "Kitchen",
    model: str = DEFAULT_TEXT_MODEL,
) -> Evaluation:
    if not constraints or not constraints.strip():
        raise AgentError("Describe your design constraints in the sidebar before curating.")
    if not materials:
        raise AgentError("There are no catalogue items to evaluate.")

    prompt = _EVAL_PROMPT.format(
        constraints=constraints.strip(),
        room_type=room_type,
        items="\n".join(m.prompt_line() for m in materials),
        categories=", ".join(CATEGORIES),
    )
    try:
        payload: _EvaluationPayload = _structured_call(
            client, model, _EVAL_SYSTEM, prompt, _EvaluationPayload, temperature=0.1
        )
    except (ValueError, ValidationError) as exc:
        raise AgentError(
            "The agent returned an unreadable evaluation. Please run the curation again.", detail=repr(exc)
        ) from exc
    except Exception as exc:
        raise friendly_api_error(exc, "evaluating materials against your constraints") from exc

    excluded = sorted({c.strip().lower() for c in payload.excluded_categories if c.strip().lower() in CATEGORIES})
    by_id = {v.id.strip(): v for v in payload.verdicts}

    evaluated: List[Material] = []
    for m in materials:
        v = by_id.get(m.id)
        if v is None:
            m.approved, m.score, m.reason = False, 0, "Not assessed by the agent — excluded for safety."
        else:
            m.approved = bool(v.approved)
            m.score = max(0, min(100, int(v.score)))
            m.reason = v.reason.strip()
            m.attributes = [a.strip() for a in v.detected_attributes if a.strip()][:6]
        # Deterministic validator: hard exclusions always win over model verdicts.
        if m.category in excluded and m.approved:
            m.approved = False
            m.reason = f"Validator override — ‘{m.category}’ is excluded by your brief. {m.reason}".strip()
        evaluated.append(m)

    evaluated.sort(key=lambda x: (not x.approved, -x.score))
    return Evaluation(
        style_summary=payload.style_summary.strip(),
        excluded_categories=excluded,
        must_haves=[s.strip() for s in payload.must_haves if s.strip()],
        materials=evaluated,
    )


# --------------------------------------------------------------------------- #
# Step 4 — Combination engine
# --------------------------------------------------------------------------- #

_PAIR_SYSTEM = """You are an award-winning interior designer specialising in compact Singapore \
HDB/BTO homes. You compose whole-room finish schemes using sound principles: a grounding base, \
a more expressive feature, value contrast, consistent colour temperature across every surface, \
texture layering and the 60-30-10 proportion rule."""

_PAIR_PROMPT = """CLIENT BRIEF: {constraints}
BRIEF SUMMARY: {summary}
MUST-HAVES: {must_haves}
ROOM: {room_type}

SURFACES TO DESIGN (surface_key | name | accepts material type | notes)
{surfaces}

APPROVED MATERIALS (id | surface type | supplier code “name” | category | finish | colour | description)
{items}

Create {n} distinct schemes. Rules:
- In each scheme assign one material id to every required surface (optional surfaces only when they
  add something). A material may only go on a surface whose type matches its own type.
- The two cabinet surfaces must use different materials: the base is calmer (solid, plaster,
  concrete), the feature carries pattern or texture (stone, fluted, textile, wood — only if allowed).
- Make every surface work together: one colour temperature, balanced contrast between cabinets,
  walls, countertop and floor, and no competing patterns side by side.
- If there are must-haves, every scheme must satisfy them.
- Vary the schemes (different moods / contrast levels); never repeat the same set of materials.
- rationale: 2 sentences explaining why the scheme works for THIS brief.
- design_principles: 2-4 short tags (e.g. "tonal layering", "warm-neutral contrast").
"""


class _Assignment(BaseModel):
    surface_key: str
    material_id: str


class _Pairing(BaseModel):
    title: str
    assignments: List[_Assignment]
    rationale: str
    design_principles: List[str]


class _PairingPayload(BaseModel):
    combinations: List[_Pairing]


_ROLE_NOTES = {"base": "calm grounding base", "feature": "expressive feature", "accent": "optional accent"}


def _ranked_for(surface: Surface, pool: List[Material]) -> List[Material]:
    """Candidates for a surface, best first, preferring suitable categories for cabinet roles."""
    ranked = sorted((m for m in pool if m.surface_type == surface.surface_type), key=lambda m: -m.score)
    if surface.role == "base":
        return [m for m in ranked if m.category in BASE_CATEGORIES] + [m for m in ranked if m.category not in BASE_CATEGORIES]
    if surface.role in ("feature", "accent"):
        return [m for m in ranked if m.category in FEATURE_CATEGORIES] + [m for m in ranked if m.category not in FEATURE_CATEGORIES]
    return ranked


def _complete_scheme(
    picks: Dict[str, Material], surfaces: List[Surface], pool: List[Material], offset: int = 0
) -> List[Tuple[Surface, Material]]:
    """Validate picks against surface types and fill any required gaps deterministically."""
    scheme: List[Tuple[Surface, Material]] = []
    used_cabinet: set = set()
    for surface in surfaces:
        m = picks.get(surface.key)
        valid = m is not None and m.surface_type == surface.surface_type and not (
            surface.surface_type == "cabinet" and m.id in used_cabinet
        )
        if not valid:
            if surface.optional:
                continue
            candidates = [c for c in _ranked_for(surface, pool)
                          if not (surface.surface_type == "cabinet" and c.id in used_cabinet)]
            if not candidates:
                continue
            m = candidates[offset % len(candidates)]
        if surface.surface_type == "cabinet":
            used_cabinet.add(m.id)
        scheme.append((surface, m))
    return scheme


def _signature(scheme: List[Tuple[Surface, Material]]) -> Tuple:
    return tuple((s.key, m.id) for s, m in scheme)


def _heuristic_combinations(surfaces: List[Surface], pool: List[Material], n: int) -> List[Combination]:
    """Deterministic fallback when the model's schemes are unusable."""
    combos: List[Combination] = []
    seen: set = set()
    for offset in range(n * 3):
        scheme = _complete_scheme({}, surfaces, pool, offset)
        if len(scheme) < 2 or _signature(scheme) in seen:
            continue
        seen.add(_signature(scheme))
        names = " × ".join(m.name for _, m in scheme[:2])
        combos.append(
            Combination(
                id=f"C{len(combos) + 1}",
                title=names,
                assignments=scheme,
                rationale=(
                    "A calm base grounds the more expressive feature, and every surface keeps to the "
                    "highest-scoring materials for your brief."
                ),
                principles=["base-feature hierarchy", "brief-compliant"],
                source="heuristic",
            )
        )
        if len(combos) >= n:
            break
    return combos


def build_combinations(
    client: genai.Client,
    evaluation: Evaluation,
    constraints: str,
    *,
    room_type: str = "Kitchen",
    surface_labels: Optional[Dict[str, Any]] = None,
    n: int = 4,
    model: str = DEFAULT_TEXT_MODEL,
    approved: Optional[List[Material]] = None,
) -> Tuple[List[Combination], List[str]]:
    """Return (combinations, warnings). ``approved`` overrides the AI shortlist.

    The surfaces come from the surface types present in the pool, so a laminate-only run
    designs the cabinets, while adding paint or countertop links extends the scheme.
    """
    pool = approved if approved is not None else evaluation.approved
    warnings: List[str] = []
    cabinet_count = sum(1 for m in pool if m.surface_type == "cabinet")
    surfaces = room_surfaces(room_type, {m.surface_type for m in pool}, surface_labels, cabinet_count)
    if len(pool) < 2 or len(surfaces) < 2:
        raise AgentError(
            "Fewer than two materials passed your constraints, so no combinations can be made. "
            "Loosen the brief, add more supplier URLs, or include items manually in the review table."
        )

    lookup = {m.id: m for m in pool}
    surface_lines = "\n".join(
        f"- {sf.key} | {sf.label} | {sf.surface_type} | "
        f"{'optional' if sf.optional else 'required'}{' · ' + _ROLE_NOTES[sf.role] if sf.role else ''}"
        for sf in surfaces
    )
    prompt = _PAIR_PROMPT.format(
        constraints=constraints.strip(),
        summary=evaluation.style_summary,
        must_haves=", ".join(evaluation.must_haves) or "none",
        room_type=room_type,
        surfaces=surface_lines,
        items="\n".join(m.prompt_line() for m in pool),
        n=n,
    )

    combos: List[Combination] = []
    try:
        payload: _PairingPayload = _structured_call(
            client, model, _PAIR_SYSTEM, prompt, _PairingPayload, temperature=0.6
        )
        seen: set = set()
        for i, p in enumerate(payload.combinations):
            picks = {a.surface_key.strip(): lookup[a.material_id.strip()]
                     for a in p.assignments if a.material_id.strip() in lookup}
            scheme = _complete_scheme(picks, surfaces, pool, offset=i)
            if len(scheme) < 2 or _signature(scheme) in seen:
                continue
            seen.add(_signature(scheme))
            combos.append(
                Combination(
                    id=f"C{len(combos) + 1}",
                    title=p.title.strip() or " × ".join(m.name for _, m in scheme[:2]),
                    assignments=scheme,
                    rationale=p.rationale.strip(),
                    principles=[t.strip() for t in p.design_principles if t.strip()][:4],
                )
            )
            if len(combos) >= n:
                break
    except AgentError:
        raise
    except (ValueError, ValidationError) as exc:
        logger.warning("Pairing payload invalid: %r", exc)
        warnings.append("The agent's pairings were unreadable — used the rule-based pairing engine instead.")
    except genai_errors.APIError as exc:
        err = friendly_api_error(exc, "generating combinations")
        if err.kind in ("auth", "model"):
            raise err from exc
        warnings.append(f"{err.message} Falling back to the rule-based pairing engine.")
    except Exception as exc:
        raise friendly_api_error(exc, "generating combinations") from exc

    if not combos:
        combos = _heuristic_combinations(surfaces, pool, n)
        if combos and not warnings:
            warnings.append("No valid AI pairings were returned — showing rule-based pairings.")
    if not combos:
        raise AgentError("Could not form any valid combinations from the approved materials.")
    return combos, warnings
