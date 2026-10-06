"""Streamlit front end for the fact checker.

app.py is the thin entry point; each module here owns one concern:

    config.py    settings (cooldown length, notice duration, widget keys)
    timing.py    pure cooldown arithmetic (no Streamlit)
    styles.py    pure CSS/HTML builders (no Streamlit)
    state.py     everything stored in st.session_state
    sidebar.py   cache controls + the fading "Cache cleared" notice
    controls.py  URL box, Analyze button, cooldown, kicking off an analysis
    analysis.py  runs the pipeline for one URL (spinners, no result drawing)
    results.py   draws a finished analysis
"""
