"""AFL Player Performance Analytics — Streamlit dashboard (entry point and navigation).

UI inspired by "Analytics Dashboard" by Lindsay (@lho), Figma Community, CC BY 4.0.
Pages live in app/views/, shared data helpers in app/common.py, layout helpers in app/ui.py.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="AFL Player Analytics", page_icon="🏉", layout="wide")

import ui  # noqa: E402
from common import ROOT, charts  # noqa: E402
from views import league, models, players, teams  # noqa: E402

charts.set_mode(st.context.theme.type == "dark")
ui.inject_css()
st.logo(str(ROOT / "app" / "static" / "logo.svg"), size="large")

pg = st.navigation({
    "League": [
        st.Page(league.page_overview, title="Season overview", icon=":material/emoji_events:", url_path="overview",
                default=True),
        st.Page(league.page_leaders, title="League leaders", icon=":material/leaderboard:", url_path="leaders"),
    ],
    "Teams": [
        st.Page(teams.page_teams, title="Team insights", icon=":material/groups:", url_path="teams"),
        st.Page(teams.page_strength, title="Team strength (Elo)", icon=":material/trending_up:", url_path="elo"),
        st.Page(teams.page_what_wins, title="What wins games", icon=":material/insights:", url_path="what-wins"),
    ],
    "Players": [
        st.Page(players.page_player, title="Player form", icon=":material/person:", url_path="player"),
        st.Page(players.page_similar, title="Similar players", icon=":material/scatter_plot:", url_path="similar"),
        st.Page(players.page_h2h, title="Head-to-head", icon=":material/compare_arrows:", url_path="head-to-head"),
    ],
    "Models": [
        st.Page(models.page_models, title="Prediction models", icon=":material/target:", url_path="models"),
        st.Page(models.page_explain, title="Why this prediction?", icon=":material/help:", url_path="explain"),
        st.Page(models.page_importance, title="Feature importance", icon=":material/bar_chart:",
                url_path="importance"),
    ],
    "About": [
        st.Page(models.page_about, title="Data & methodology", icon=":material/menu_book:", url_path="about"),
    ],
})
st.sidebar.caption("Data: AFL Tables (player stats) · Squiggle API (results cross-check). Seasons 2021–2026.  \n"
                   "UI inspired by *Analytics Dashboard* by Lindsay (@lho), Figma Community, CC BY 4.0.")
pg.run()
