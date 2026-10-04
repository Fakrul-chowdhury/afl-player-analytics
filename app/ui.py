"""Layout helpers for the card-based look (styles in app/style.css, colours in .streamlit/config.toml).

UI inspired by "Analytics Dashboard" by Lindsay (@lho), Figma Community, CC BY 4.0.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

CSS = Path(__file__).with_name("style.css")


# Extra design tokens per theme (everything else comes from .streamlit/config.toml).
TOKENS = {
    "light": {"card": "#ffffff", "card-border": "#e9edf4", "hairline": "#e4e8f0", "ink": "#1d2433",
              "muted": "#6b7385", "accent": "#2f5bc9", "nav-active": "#eef3fe",
              "shadow": "0 1px 2px rgba(16,24,40,.04), 0 4px 14px rgba(16,24,40,.05)",
              "shadow-sm": "0 1px 2px rgba(16,24,40,.06)", "shadow-side": "1px 0 0 #e9edf4"},
    "dark": {"card": "#1c2030", "card-border": "#2b3144", "hairline": "#2b3144", "ink": "#e6e9f2",
             "muted": "#a3abbd", "accent": "#8fb0ff", "nav-active": "#232b45",
             "shadow": "0 1px 2px rgba(0,0,0,.35), 0 6px 18px rgba(0,0,0,.25)",
             "shadow-sm": "0 1px 2px rgba(0,0,0,.4)", "shadow-side": "1px 0 0 #2b3144"},
}


def inject_css() -> None:
    mode = "dark" if st.context.theme.type == "dark" else "light"
    tokens = ";".join(f"--{k}:{v}" for k, v in TOKENS[mode].items())
    st.html(f"<style>:root{{{tokens}}}\n{CSS.read_text(encoding='utf-8')}</style>")


def header(title: str, subtitle: str | None = None) -> None:
    """Page title row with a muted subtitle and a hairline divider, as in the design's report header."""
    st.markdown(f"<div class='page-head'><h1>{title}</h1>"
                + (f"<p>{subtitle}</p>" if subtitle else "") + "</div>", unsafe_allow_html=True)


def card(title: str | None = None, caption: str | None = None, key: str | None = None):
    """A white, rounded, shadowed tile. Use as a context manager."""
    box = st.container(border=True, key=key)
    # The marker div doubles as the title; an untitled card gets an empty marker that CSS hides.
    box.markdown(f"<div class='card-marker card-title'>{title or ''}</div>", unsafe_allow_html=True)
    if caption:
        box.caption(caption)
    return box


def filter_bar():
    """A row of filter controls above the content (the design's dropdown bar)."""
    return st.container(key="filter_bar")


def kpi(col, label: str, value, delta=None, *, help: str | None = None, spark=None, neutral: bool = False,
        inverse: bool = False) -> None:
    """KPI tile: label, big number, optional delta and an optional sparkline of real values."""
    kw = {"delta_color": "off", "delta_arrow": "off"} if neutral else (
        {"delta_color": "inverse"} if inverse else {})
    col.metric(label, value, delta, border=True, help=help,
               chart_data=list(spark) if spark is not None else None, chart_type="area", **kw)
