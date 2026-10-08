"""The "👋 Start here" landing page: a hero banner, step cards, and a call-to-action.

Pure presentation - edit the content in app.py, not here.
"""
import streamlit as st

CSS = """
<style>
.tour-hero {border-radius: 20px; padding: 30px 34px; color: #fff; margin-bottom: 18px;}
.tour-kicker {font-size: 13px; letter-spacing: .12em; text-transform: uppercase; opacity: .85;}
.tour-title {font-size: 40px; font-weight: 800; line-height: 1.1; margin: 6px 0 8px;}
.tour-sub {font-size: 18px; opacity: .95; max-width: 760px;}
.tour-card {border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px 18px 14px; height: 100%;
            background: #fff; box-shadow: 0 1px 2px rgba(15,23,42,.04);}
.tour-num {display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px;
           border-radius: 999px; color: #fff; font-weight: 700; font-size: 14px; margin-right: 8px;}
.tour-emoji {font-size: 30px; margin: 10px 0 4px;}
.tour-card h4 {margin: 2px 0 6px; font-size: 17px;}
.tour-card p {font-size: 14px; color: #334155; margin: 0 0 10px;}
.tour-hood {font-size: 12.5px; color: #475569; background: #f1f5f9; border-radius: 10px; padding: 7px 9px;}
.tour-chip {display: inline-block; padding: 4px 10px; margin: 0 6px 6px 0; border-radius: 999px;
            background: #f1f5f9; border: 1px solid #e2e8f0; font-size: 13px; color: #0f172a;}
</style>
"""


def render(kicker: str, title: str, subtitle: str, gradient: tuple[str, str], steps: list[dict],
           chips: list[str], accent: str) -> None:
    """steps: [{"emoji", "title", "text", "hood"}, ...]"""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f"<div class='tour-hero' style='background:linear-gradient(135deg,{gradient[0]},{gradient[1]})'>"
        f"<div class='tour-kicker'>{kicker}</div><div class='tour-title'>{title}</div>"
        f"<div class='tour-sub'>{subtitle}</div></div>", unsafe_allow_html=True)

    st.markdown("#### How it works, in steps")
    for row in range(0, len(steps), 3):
        cols = st.columns(3)
        for i, (col, s) in enumerate(zip(cols, steps[row:row + 3]), start=row + 1):
            col.markdown(
                f"<div class='tour-card'><span class='tour-num' style='background:{accent}'>{i}</span>"
                f"<div class='tour-emoji'>{s['emoji']}</div><h4>{s['title']}</h4><p>{s['text']}</p>"
                f"<div class='tour-hood'>🔧 <b>Under the hood:</b> {s['hood']}</div></div>",
                unsafe_allow_html=True)
        st.write("")

    st.markdown("#### Built with")
    st.markdown("".join(f"<span class='tour-chip'>{c}</span>" for c in chips), unsafe_allow_html=True)
    st.write("")
