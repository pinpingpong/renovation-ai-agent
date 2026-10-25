"""
Renovation AI Agent — Streamlit frontend.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (set GEMINI_API_KEY in app secrets)
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from typing import List, Optional, Tuple

from urllib.parse import urlparse

import pandas as pd
import streamlit as st

import agent_logic as agent
import renderer
import showcase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("renovation_ai")

st.set_page_config(
    page_title="Renovation AI Agent",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

AUTO_DETECT = "Auto-detect"
SURFACE_SHORT = {"cabinet": "Cabinets", "wall": "Walls", "countertop": "Countertop",
                 "backsplash": "Backsplash", "floor": "Floor"}
SURFACE_TYPE_OPTIONS = [AUTO_DETECT] + list(SURFACE_SHORT.values())
SURFACE_TYPE_KEY = {label: key for key, label in SURFACE_SHORT.items()}
SURFACE_ICON = {"cabinet": "🗄️", "wall": "🧱", "countertop": "🍳", "backsplash": "🔲", "floor": "🟫"}

BRIEF_PRESETS = {
    "Desert Modernism · zero wood (BaZi)": (
        "Desert modernism for a 5-room BTO in Singapore. Absolutely ZERO wood or timber-look "
        "laminates (BaZi elemental rule). Favour sun-bleached plaster, limestone, travertine and "
        "sandstone patterns, plus warm terracotta-taupe and sand earth tones. Matte, suede or "
        "textured finishes. Avoid high gloss, cool blue-greys and stark white."
    ),
    "Japandi · light oak + fluting": (
        "Japandi style. Must include at least one light oak (or ash) woodgrain, ideally paired with "
        "vertical fluting or a linear ribbed texture. Muted warm whites, greige and soft charcoal "
        "accents. Matte finishes only. Avoid marble veining and anything high gloss."
    ),
    "Quiet luxury · stone + dark solids": (
        "Quiet-luxury apartment. Favour honed stone looks (travertine, warm marble) against deep "
        "solid colours such as espresso, olive or charcoal. Minimal patterns, no high gloss, "
        "no bright or saturated colours."
    ),
    "Custom (write your own)": "",
}

SAMPLE_URLS = (
    "https://www.lamitak.com/collections/solids\n"
    "https://www.lamitak.com/collections/patterns"
)

CATEGORY_ICON = {
    "solid": "⬜", "stone": "🪨", "plaster": "🧱", "concrete": "🏗️", "terrazzo": "🔘",
    "wood": "🪵", "textile": "🧵", "metallic": "⚙️", "fluted": "〰️", "pattern": "🔷", "other": "▫️",
}

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1280px;}
      .hero h1 {margin-bottom: .25rem;}
      .hero p {color: var(--text-color); opacity: .75; font-size: 1.05rem; margin-top: 0;}
      .step {font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
             color: #B5653B; margin-bottom: -.4rem;}
      .chip {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
             margin: 2px 4px 2px 0; background: rgba(181,101,59,.12); color: #8A4A2A;}
      .chip.bad {background: rgba(170,40,40,.12); color: #9A2A2A;}
      .swatch-label {font-size: .8rem; line-height: 1.25; margin-top: -.35rem;}
      .muted {opacity: .7; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

DEFAULT_STATE = {
    "catalog": None,          # agent.CatalogResult
    "evaluation": None,       # agent.Evaluation
    "combinations": [],       # List[agent.Combination]
    "pipeline_warnings": [],
    "selected_combo": None,   # combination id
    "renders": [],            # List[renderer.RenderResult]
    "base_image_png": None,   # bytes of the prepared base image
    "base_image_hash": None,
    "demo_runs": 0,           # curation runs this session that used the shared demo key
    "run_meta": None,         # brief/urls/models of the last curation run (for showcase export)
    "brief_preset": list(BRIEF_PRESETS)[0],
    "constraints": list(BRIEF_PRESETS.values())[0],
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)


def _reset_results() -> None:
    for k in ("catalog", "evaluation", "selected_combo"):
        st.session_state[k] = None
    st.session_state["combinations"] = []
    st.session_state["pipeline_warnings"] = []


def _apply_brief_preset() -> None:
    text = BRIEF_PRESETS.get(st.session_state["brief_preset"], "")
    if text:
        st.session_state["constraints"] = text


def show_error(err: agent.AgentError) -> None:
    st.error(err.message, icon="⚠️")
    if err.detail:
        with st.expander("Technical details"):
            st.code(err.detail, language="text")


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


def _configured_secret(name: str, *env_names: str) -> Optional[str]:
    """Look up a key in Streamlit secrets (cloud) then the environment (local).

    Placeholder values copied from secrets.toml.example are ignored.
    """
    def usable(v: Optional[str]) -> bool:
        return bool(v) and not v.lower().startswith("your-") and "..." not in v

    try:
        if name in st.secrets and usable(str(st.secrets[name]).strip()):
            return str(st.secrets[name]).strip()
    except Exception:  # no secrets.toml present
        pass
    for env in (name, *env_names):
        value = os.environ.get(env, "").strip()
        if usable(value):
            return value
    return None


def _configured_api_key() -> Optional[str]:
    return _configured_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")


# Owner's free-tier key (a Google project WITHOUT billing) that keyless visitors share
# for Steps 1–4. It is never used for image generation.
DEMO_RUNS_PER_SESSION = 3


def _count_demo_run() -> None:
    """Button callback: runs before the rerun, so the sidebar counter is never stale."""
    st.session_state["demo_runs"] += 1


# --------------------------------------------------------------------------- #
# Cached agent calls
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def cached_resolve(
    urls: Tuple[str, ...], types: Tuple[str, ...], model: str, max_items: int, _api_key: str
) -> agent.CatalogResult:
    """Catalogue data is public, so it is cached across sessions by URL + surface type + model."""
    client = agent.make_client(_api_key)
    return agent.resolve_catalog(
        client, urls, model=model, max_items_per_url=max_items, surface_types=dict(zip(urls, types))
    )


@st.cache_data(ttl=3600, show_spinner=False, max_entries=16)
def cached_models(key_fingerprint: str, _api_key: str) -> Tuple[List[str], List[str]]:
    """Models depend on the key, so the cache is keyed by a hash of it (never the key itself)."""
    return agent.list_generation_models(agent.make_client(_api_key))


def _model_picker(label: str, available: List[str], preferred, fallback: str, help_text: str) -> str:
    if not available:
        return st.text_input(label, fallback, help=help_text)
    default = agent.pick_default_model(available, preferred) or available[0]
    return st.selectbox(label, available, index=available.index(default), help=help_text)


@st.cache_data(show_spinner=False, max_entries=256)
def swatch_preview(supplier: str, code: str, category: str, hex_color: str) -> bytes:
    m = agent.Material(id="preview", code=code, name=code, supplier=supplier, category=category, hex_color=hex_color)
    buf = io.BytesIO()
    renderer.synthesise_swatch(m, side=160).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Configuration")

    configured_key = _configured_api_key()
    typed_key = st.text_input(
        "Gemini API key",
        type="password",
        placeholder="Using configured key" if configured_key else "Paste your key (AIza…)",
        help="Stored only in this browser session. On Streamlit Cloud, set GEMINI_API_KEY in app secrets.",
    )
    user_key = (typed_key or configured_key or "").strip()
    demo_key = _configured_secret("GEMINI_DEMO_KEY") or ""
    using_demo = not user_key and bool(demo_key)
    # Steps 1–4 can fall back to the shared demo key; rendering only ever uses the user's key.
    api_key = user_key or demo_key
    demo_runs_left = DEMO_RUNS_PER_SESSION - st.session_state["demo_runs"]

    if user_key:
        source = "sidebar" if typed_key else "secrets / environment"
        st.caption(f"✅ Your API key loaded from {source}. All steps, including rendering, are available.")
    elif using_demo:
        st.info(
            f"**Using the shared demo key.** You can curate catalogues and get combination "
            f"suggestions ({max(demo_runs_left, 0)} of {DEMO_RUNS_PER_SESSION} runs left this session), "
            "but **no images will be rendered**. Paste your own key above to render.",
            icon="🔑",
        )
    else:
        st.caption("Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).")
    with st.expander("ℹ️ About keys & costs", expanded=not user_key):
        st.markdown(
            "- **Bring your own Gemini API key to render.** It stays in your browser session and is never stored.\n"
            "- **No key?** Steps 1–4 (catalogue lookup, curation and pairing) run on a shared free-tier "
            "demo key, limited to a few runs per visit. It can hit its daily quota when many people "
            "use it, and Google may use free-tier requests to improve its products, so don't "
            "upload private photos while using the demo key.\n"
            "- **Image rendering (Step 5)** uses Gemini image models, which are **paid-only**: "
            "free-tier keys have 0 image quota. To render, use your own key from a Google project with "
            "[billing enabled](https://aistudio.google.com/apikey). A render typically costs a few cents."
        )

    st.divider()
    st.subheader("🎨 Design brief")
    st.selectbox("Start from a preset", list(BRIEF_PRESETS), key="brief_preset", on_change=_apply_brief_preset)
    st.text_area(
        "Your aesthetic rules",
        key="constraints",
        height=210,
        help="Write freely. Negative rules (“zero wood”) are enforced as hard exclusions; "
        "positive rules (“must include oak”) guide scoring and pairing.",
        placeholder="e.g. Japandi style, must include light oak wood and vertical fluting…",
    )

    st.divider()
    st.subheader("🏠 Room")
    room_type = st.selectbox(
        "Room type", list(agent.ROOM_SURFACE_LABELS),
        help="The surfaces to redesign come from your supplier links: laminates → cabinets, "
             "paint → walls, quartz → countertop, and so on.",
    )
    defaults = agent.ROOM_SURFACE_LABELS[room_type]
    with st.expander("Rename surfaces (optional)"):
        cab = defaults["cabinet"]
        cab_names = tuple(
            st.text_input(lbl, value, key=f"cab{i}_{room_type}") or value
            for i, (lbl, value) in enumerate(zip(("Cabinet base", "Cabinet feature", "Cabinet accent"), cab))
        )
        surface_labels = {"cabinet": cab_names}
        for t in ("wall", "countertop", "backsplash", "floor"):
            surface_labels[t] = st.text_input(agent.SURFACE_TYPES[t], defaults[t], key=f"{t}_{room_type}") or defaults[t]

    text_models: List[str] = []
    image_models: List[str] = []
    if api_key:
        try:
            text_models, image_models = cached_models(hashlib.sha256(api_key.encode()).hexdigest(), api_key)
        except agent.AgentError as err:
            st.caption(f"⚠️ Couldn't list your models: {err.message}")

    with st.expander("Advanced model settings"):
        text_model = _model_picker(
            "Reasoning / grounding model", text_models, agent.PREFERRED_TEXT_MODELS,
            agent.DEFAULT_TEXT_MODEL, "Used for search-grounded catalogue lookup, evaluation and pairing.",
        )
        image_model = _model_picker(
            "Image model", image_models, renderer.PREFERRED_IMAGE_MODELS,
            renderer.DEFAULT_IMAGE_MODEL, "Used for multimodal inpainting of your room photo.",
        )
        if api_key and text_models:
            st.caption(f"Showing {len(text_models)} text and {len(image_models)} image models available to your key.")
        max_items = st.slider("Max products per URL", 8, 40, 20, step=4)
        n_combos = st.slider("Combinations to propose", 2, 8, 4)
        test_image = st.checkbox("Include image model in test", value=True,
                                 help="Uses one small image generation request.")
        if st.button("🩺 Test key & models", width="stretch", disabled=not user_key,
                     help="Sends tiny requests: plain, Google Search grounding, and optionally one image."):
            with st.spinner(f"Testing {text_model}{' and ' + image_model if test_image else ''}…"):
                checks = agent.check_model_access(
                    agent.make_client(user_key), text_model, image_model if test_image else None
                )
                for label, ok, message in checks:
                    (st.success if ok else st.error)(f"**{label}:** {'works' if ok else 'failed'}")
                    if not ok:
                        st.code(message[:800], language="text", wrap_lines=True)

    st.divider()
    if st.button("↺ Reset session", width="stretch"):
        _reset_results()
        st.session_state["renders"] = []
        st.rerun()

constraints: str = st.session_state["constraints"]

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="hero">
      <h1>🏠 Renovation AI Agent</h1>
      <p>An autonomous interior-design copilot: paste real supplier catalogues, describe your rules,
      and preview curated laminate combinations on <b>your own room photo</b> — geometry preserved.</p>
    </div>
    """,
    unsafe_allow_html=True,
)



@st.cache_data(show_spinner=False, max_entries=2)
def _load_showcase(version: float) -> Optional[showcase.Showcase]:
    """``version`` is the manifest's mtime, so a newly committed showcase appears without a restart."""
    return showcase.load_showcase()


def _swatch_chip(m: dict) -> str:
    hex_color = m.get("hex_color") or "#CCCCCC"
    return (f"<span class='chip'><span style='display:inline-block;width:.8em;height:.8em;border-radius:2px;"
            f"background:{hex_color};margin-right:4px;vertical-align:-1px'></span>"
            f"{m.get('code', '')} {m.get('name', '')}</span>")


demo = _load_showcase(showcase.showcase_version())
if demo is not None and (demo.renders or demo.combinations):
    with st.expander("✨ See a real end-to-end example (no API key needed)", expanded=not user_key):
        meta = demo.meta
        st.markdown(f"**Brief:** {meta.get('brief', '')}")
        st.caption(
            f"{meta.get('room_type', '')} · {len(demo.materials)} materials resolved from "
            f"{len(meta.get('urls', []))} supplier URL(s) · models: `{meta.get('text_model', '')}` + "
            f"`{meta.get('image_model', '')}` · run on {meta.get('created', '')}"
        )
        for r in demo.renders:
            st.markdown(f"**{r.title}**")
            b, a = st.columns(2, gap="small")
            if demo.before:
                b.image(demo.before, caption="Before — base photo", width="stretch")
            a.image(r.image, caption="After — AI render", width="stretch")
            if r.swatches:
                st.caption(" · ".join(label for label, _ in r.swatches))
        if demo.combinations:
            st.markdown("**Combinations the agent proposed**")
            for c in demo.combinations:
                chips = "".join(f"<span class='muted'>{a.get('surface', '')}:</span> " + _swatch_chip(a.get("material") or {})
                                for a in c.get("assignments", []))
                st.markdown(f"**{c.get('title', '')}** {chips}<br><span class='muted'>{c.get('rationale', '')}</span>",
                            unsafe_allow_html=True)
        if demo.materials:
            with st.popover("📋 Full curated material list"):
                st.dataframe(
                    pd.DataFrame(demo.materials)[["approved", "code", "name", "supplier", "category", "score", "reason"]]
                    .rename(columns={"approved": "Approved", "score": "Fit", "reason": "Agent reasoning"}),
                    hide_index=True, width="stretch",
                )

progress_cols = st.columns(5)
milestones = [
    ("Room photo", st.session_state["base_image_png"] is not None),
    ("Catalogue URLs", False),  # filled below once parsed
    ("Curated", st.session_state["evaluation"] is not None),
    ("Combination picked", st.session_state["selected_combo"] is not None),
    ("Rendered", bool(st.session_state["renders"])),
]

# --------------------------------------------------------------------------- #
# Step 1 & 2 — Inputs
# --------------------------------------------------------------------------- #

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="step">Step 1</div>', unsafe_allow_html=True)
    st.subheader("Upload your base room photo")
    upload = st.file_uploader(
        "Kitchen, bedroom, living room…",
        type=["jpg", "jpeg", "png", "webp"],
        help="A straight-on, well-lit photo or 3D perspective works best. Max 15 MB.",
    )
    if upload is not None:
        raw = upload.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != st.session_state["base_image_hash"]:
            try:
                img = renderer.prepare_base_image(raw)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.session_state["base_image_png"] = buf.getvalue()
                st.session_state["base_image_hash"] = digest
            except agent.AgentError as err:
                st.session_state["base_image_png"] = None
                show_error(err)
    else:
        st.session_state["base_image_png"] = None
        st.session_state["base_image_hash"] = None

    if st.session_state["base_image_png"]:
        st.image(st.session_state["base_image_png"], caption="Base room", width="stretch")

with right:
    st.markdown('<div class="step">Step 2</div>', unsafe_allow_html=True)
    st.subheader("Supplier portfolio URLs")
    url_text = st.text_area(
        "One URL per line (Lamitak, Jennings, EDL, Admira…)",
        key="url_text",
        height=130,
        placeholder=SAMPLE_URLS,
        help=f"Up to {agent.MAX_URLS} collection or product-listing pages.",
    )
    urls, url_problems = agent.parse_urls(url_text)
    for p in url_problems if url_text.strip() else []:
        st.caption(f"⚠️ {p}")
    url_types: dict = {}
    if urls:
        st.caption("**What is each link for?** Leave on Auto-detect and the agent works it out.")
        for u in urls:
            c1, c2 = st.columns([1, 1], vertical_alignment="center")
            c1.markdown(f"[{urlparse(u).hostname}]({u})<br><span class='muted'>{urlparse(u).path[:48]}</span>",
                        unsafe_allow_html=True)
            choice = c2.selectbox("Used for", SURFACE_TYPE_OPTIONS, key=f"type_{hashlib.md5(u.encode()).hexdigest()[:10]}",
                                  label_visibility="collapsed")
            url_types[u] = SURFACE_TYPE_KEY.get(choice, "auto")
    milestones[1] = ("Catalogue URLs", bool(urls))

    st.markdown('<div class="step">Step 3</div>', unsafe_allow_html=True)
    st.subheader("Fetch & curate")
    missing = []
    if not api_key:
        missing.append("API key")
    elif using_demo and demo_runs_left <= 0:
        missing.append("your own API key (demo runs used up for this session)")
    if not urls:
        missing.append("at least one URL")
    if not constraints.strip():
        missing.append("a design brief")
    run_pipeline = st.button(
        "🔎 Fetch & Curate Catalog",
        type="primary",
        width="stretch",
        disabled=bool(missing),
        help=("Needs: " + ", ".join(missing)) if missing else "Resolve → evaluate → pair",
        on_click=_count_demo_run if using_demo else None,
    )
    if missing:
        st.caption("Needs " + ", ".join(missing) + ".")

for col, (label, done) in zip(progress_cols, milestones):
    col.markdown(f"🟠 **{label}**" if done else f"⚪ <span class='muted'>{label}</span>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Pipeline execution
# --------------------------------------------------------------------------- #

if run_pipeline and using_demo and st.session_state["demo_runs"] > DEMO_RUNS_PER_SESSION:
    st.warning("You've used this session's demo runs. Paste your own Gemini API key in the sidebar to continue.")
    run_pipeline = False

if run_pipeline:
    _reset_results()
    st.session_state["run_meta"] = {
        "brief": constraints, "urls": list(urls), "room_type": room_type,
        "text_model": text_model,
    }
    try:
        client = agent.make_client(api_key)
        with st.spinner(f"Resolving {len(urls)} catalogue link(s) with Gemini Search grounding…"):
            catalog = cached_resolve(
                tuple(urls), tuple(url_types.get(u, "auto") for u in urls), text_model, max_items, api_key
            )
            st.session_state["catalog"] = catalog
        with st.spinner(f"Evaluating {len(catalog.materials)} materials against your brief…"):
            # Work on copies so the cached catalogue is never mutated.
            fresh = [agent.Material(**{**m.__dict__, "attributes": list(m.attributes)}) for m in catalog.materials]
            evaluation = agent.evaluate_materials(
                client, fresh, constraints, room_type=room_type, model=text_model
            )
            st.session_state["evaluation"] = evaluation
        with st.spinner("Composing whole-room schemes…"):
            combos, warns = agent.build_combinations(
                client, evaluation, constraints,
                room_type=room_type, surface_labels=surface_labels, n=n_combos, model=text_model,
            )
            st.session_state["combinations"] = combos
            st.session_state["pipeline_warnings"] = list(catalog.warnings) + warns
            st.session_state["selected_combo"] = combos[0].id if combos else None
        st.toast(f"Curated {len(evaluation.approved)} materials into {len(combos)} combinations", icon="✅")
    except agent.AgentError as err:
        show_error(err)
    except Exception as exc:  # last-resort guard — never show a raw traceback
        logger.exception("Pipeline failure")
        show_error(agent.friendly_api_error(exc, "running the curation pipeline"))

# --------------------------------------------------------------------------- #
# Step 4 — Review grid
# --------------------------------------------------------------------------- #

evaluation: Optional[agent.Evaluation] = st.session_state["evaluation"]
catalog: Optional[agent.CatalogResult] = st.session_state["catalog"]


def _swatch(m: agent.Material, width: int = 110) -> None:
    if m.swatch_image_url:
        st.image(m.swatch_image_url, width=width)
    else:
        st.image(swatch_preview(m.supplier, m.code, m.category, m.hex_color), width=width)
    st.markdown(
        f"<div class='swatch-label'><b>{m.code}</b> {m.name}<br>"
        f"<span class='muted'>{CATEGORY_ICON.get(m.category, '')} {m.category} · {m.supplier}</span></div>",
        unsafe_allow_html=True,
    )


if evaluation is not None:
    st.divider()
    st.markdown('<div class="step">Step 4</div>', unsafe_allow_html=True)
    st.subheader("Review the curated combinations")

    summary_cols = st.columns([3, 1, 1, 1])
    summary_cols[0].markdown(f"**Brief understood as:** {evaluation.style_summary}")
    chips = "".join(f"<span class='chip bad'>✕ {c}</span>" for c in evaluation.excluded_categories)
    chips += "".join(f"<span class='chip'>✓ {m}</span>" for m in evaluation.must_haves)
    if chips:
        summary_cols[0].markdown(chips, unsafe_allow_html=True)
    summary_cols[1].metric("Resolved", len(evaluation.materials))
    summary_cols[2].metric("Approved", len(evaluation.approved))
    summary_cols[3].metric("Combinations", len(st.session_state["combinations"]))

    for w in st.session_state["pipeline_warnings"]:
        st.warning(w, icon="ℹ️")

    tab_combos, tab_materials, tab_sources = st.tabs(["🧩 Combinations", "📋 Material review", "🔗 Grounding sources"])

    # ---- Combinations grid ------------------------------------------------- #
    with tab_combos:
        combos: List[agent.Combination] = st.session_state["combinations"]
        if not combos:
            st.info("No combinations yet — adjust the material selection and rebuild.")
        for row_start in range(0, len(combos), 2):
            cols = st.columns(2, gap="medium")
            for col, combo in zip(cols, combos[row_start : row_start + 2]):
                selected = combo.id == st.session_state["selected_combo"]
                with col.container(border=True):
                    head = st.columns([4, 1])
                    head[0].markdown(f"#### {'✅ ' if selected else ''}{combo.title}")
                    if combo.source == "heuristic":
                        head[1].caption("rule-based")
                    n_cols = max(3, len(combo.assignments))
                    sw_cols = st.columns(n_cols)
                    for sc, (surface, mat) in zip(sw_cols, combo.assignments):
                        with sc:
                            st.caption(f"{SURFACE_ICON.get(surface.surface_type, '')} **{surface.label}**")
                            _swatch(mat, width=110 if n_cols <= 3 else 80)
                    st.markdown(combo.rationale)
                    if combo.principles:
                        st.markdown("".join(f"<span class='chip'>{p}</span>" for p in combo.principles),
                                    unsafe_allow_html=True)
                    if st.button(
                        "Selected" if selected else "Select this combination",
                        key=f"pick_{combo.id}",
                        type="primary" if selected else "secondary",
                        width="stretch",
                        disabled=selected,
                    ):
                        st.session_state["selected_combo"] = combo.id
                        st.rerun()

    # ---- Material review / override --------------------------------------- #
    with tab_materials:
        st.caption(
            "The agent's verdicts are pre-filled. Tick or untick **Include** to override them, "
            "then rebuild the combinations from your own shortlist."
        )
        df = pd.DataFrame(
            [
                {
                    "Include": bool(m.approved),
                    "Swatch": m.swatch_image_url or None,
                    "Code": m.code,
                    "Name": m.name,
                    "For": f"{SURFACE_ICON.get(m.surface_type, '')} {m.surface_type}",
                    "Supplier": m.supplier,
                    "Category": f"{CATEGORY_ICON.get(m.category, '')} {m.category}",
                    "Finish": m.finish,
                    "Colour": f"{m.color_family} {m.hex_color}".strip(),
                    "Fit": m.score,
                    "Agent reasoning": m.reason,
                    "Link": m.product_url or m.source_url,
                    "_id": m.id,
                }
                for m in evaluation.materials
            ]
        )
        edited = st.data_editor(
            df,
            key="material_editor",
            hide_index=True,
            width="stretch",
            height=min(560, 42 + 36 * len(df)),
            disabled=[c for c in df.columns if c != "Include"],
            column_order=[c for c in df.columns if c != "_id"],
            column_config={
                "Include": st.column_config.CheckboxColumn(width="small"),
                "Swatch": st.column_config.ImageColumn(width="small"),
                "Fit": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d", width="small"),
                "Agent reasoning": st.column_config.TextColumn(width="large"),
                "Link": st.column_config.LinkColumn(display_text="open", width="small"),
            },
        )
        chosen_ids = set(edited.loc[edited["Include"], "_id"])
        overridden = chosen_ids != {m.id for m in evaluation.approved}
        rebuild = st.button(
            f"🔁 Rebuild combinations from {len(chosen_ids)} selected materials",
            disabled=len(chosen_ids) < 2 or not api_key or (using_demo and demo_runs_left <= 0),
            type="primary" if overridden else "secondary",
            on_click=_count_demo_run if using_demo else None,
        )
        if rebuild:
            pool = [m for m in evaluation.materials if m.id in chosen_ids]
            try:
                with st.spinner("Re-composing pairings from your shortlist…"):
                    combos, warns = agent.build_combinations(
                        agent.make_client(api_key), evaluation, constraints,
                        room_type=room_type, surface_labels=surface_labels, n=n_combos,
                        model=text_model, approved=pool,
                    )
                st.session_state["combinations"] = combos
                st.session_state["pipeline_warnings"] = warns
                st.session_state["selected_combo"] = combos[0].id
                st.rerun()
            except agent.AgentError as err:
                show_error(err)
            except Exception as exc:
                show_error(agent.friendly_api_error(exc, "rebuilding combinations"))

    # ---- Sources ----------------------------------------------------------- #
    with tab_sources:
        if catalog is not None:
            st.markdown("**Per-URL resolution**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "URL": r.url,
                            "Supplier": r.supplier,
                            "Collection": r.collection,
                            "Used for": (agent.SURFACE_TYPES.get(r.surface_type, "—")
                                         + (" (auto-detected)" if r.detected and r.surface_type else "")),
                            "Items": r.items_found,
                            "Server reachable": "yes" if r.reachable else "no",
                            "Note": r.note,
                        }
                        for r in catalog.reports
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={"URL": st.column_config.LinkColumn()},
            )
            if catalog.sources:
                st.markdown("**Grounding citations returned by Gemini**")
                for s in catalog.sources[:40]:
                    st.markdown(f"- [{s['title']}]({s['uri']}) <span class='muted'>· {s['via']}</span>",
                                unsafe_allow_html=True)
            else:
                st.caption("No grounding citations were returned for this run.")

# --------------------------------------------------------------------------- #
# Step 5 — Render
# --------------------------------------------------------------------------- #

combos = st.session_state["combinations"]
selected: Optional[agent.Combination] = next(
    (c for c in combos if c.id == st.session_state["selected_combo"]), None
)

if selected is not None:
    st.divider()
    st.markdown('<div class="step">Step 5</div>', unsafe_allow_html=True)
    st.subheader(f"Render “{selected.title}” onto your room")

    surface_names = {sf.key: sf.label for sf, _ in selected.assignments}
    chosen_surfaces = st.multiselect(
        "Surfaces to change", list(surface_names), default=list(surface_names),
        format_func=lambda k: surface_names[k],
        help="Untick a surface to keep it as it is in your photo.",
        key=f"surfaces_{selected.id}",
    )
    extra = st.text_input(
        "Extra rendering notes (optional)",
        placeholder="e.g. keep the existing quartz countertop; brushed brass handles stay as-is",
    )

    if user_key:
        st.caption(
            f"🎨 Rendering uses `{image_model}`. Gemini image models are paid-only, so your key's "
            "Google project needs billing enabled (a render typically costs a few cents)."
        )
    else:
        st.info(
            "**Rendering needs your own Gemini API key.** The shared demo key covers catalogue curation "
            "and combination suggestions only. Gemini image models are paid-only, so paste a key from a "
            "Google project with billing enabled in the sidebar to render this combination onto your room.",
            icon="🎨",
        )
    can_render = st.session_state["base_image_png"] is not None and bool(user_key) and bool(chosen_surfaces)
    if st.session_state["base_image_png"] is None:
        st.info("Upload a base room photo in Step 1 to enable rendering.", icon="🖼️")

    if st.button("🎨 Generate render", type="primary", disabled=not can_render, width="stretch"):
        try:
            base_img = renderer.prepare_base_image(st.session_state["base_image_png"])
            with st.spinner(f"Fetching swatches and inpainting with {image_model}… (20–60 s)"):
                result = renderer.render_combination(
                    agent.make_client(user_key), base_img, selected,
                    room_type=room_type, model=image_model,
                    extra_instructions=extra, surface_keys=chosen_surfaces,
                )
            st.session_state["renders"].insert(0, result)
            st.toast("Render complete", icon="🎨")
        except agent.AgentError as err:
            show_error(err)
        except Exception as exc:
            logger.exception("Render failure")
            show_error(agent.friendly_api_error(exc, "rendering"))

# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #

renders: List[renderer.RenderResult] = st.session_state["renders"]
if renders:
    st.divider()
    st.subheader("🖼️ Output gallery")
    for i, r in enumerate(renders):
        with st.container(border=True):
            st.markdown(f"**{r.combination_title}**  <span class='muted'>· render {len(renders) - i}</span>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="small")
            if st.session_state["base_image_png"]:
                c1.image(st.session_state["base_image_png"], caption="Before", width="stretch")
            c2.image(r.image_bytes, caption="After", width="stretch")
            meta = " · ".join(f"{label} ({origin} swatch)" for label, origin in r.swatch_origins)
            st.caption(meta)
            if r.model_note:
                st.caption(f"Model note: {r.model_note[:300]}")
            ext = "png" if "png" in r.mime_type else "jpg"
            st.download_button(
                "⬇️ Download render",
                data=r.image_bytes,
                file_name=f"renovation_{r.combination_id}_{len(renders) - i}.{ext}",
                mime=r.mime_type,
                key=f"dl_{i}",
            )

if renders and st.session_state["evaluation"] is not None:
    run = st.session_state["run_meta"] or {}
    with st.expander("💾 Save this run as the app's showcase"):
        st.markdown(
            "Downloads a zip containing a `showcase/` folder. Unzip it into the project root and commit it. "
            "The app then shows this run to every visitor under **See a real end-to-end example**, "
            "with no API calls."
        )
        st.download_button(
            "⬇️ Download showcase.zip",
            data=showcase.build_showcase_zip(
                brief=run.get("brief", constraints),
                room_type=run.get("room_type", room_type),
                urls=run.get("urls", urls),
                text_model=run.get("text_model", text_model),
                image_model=image_model,
                evaluation=st.session_state["evaluation"],
                combinations=st.session_state["combinations"],
                renders=renders,
                base_png=st.session_state["base_image_png"],
            ),
            file_name="showcase.zip",
            mime="application/zip",
        )

st.markdown("---")
st.caption(
    "Renovation AI Agent · Gemini Search Grounding + multimodal inpainting via `google-genai`. "
    "Product data is AI-resolved from public supplier pages — verify codes with your supplier before ordering."
)
