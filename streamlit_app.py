import streamlit as st


st.set_page_config(
    page_title="Steam User Reception Explorer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


pages = {
    "Steam Reception Explorer": [
        st.Page(
            "app/pages/overview.py",
            title="Overview",
            icon="🏠",
            default=True,
        ),
        st.Page(
            "app/pages/game_explorer.py",
            title="Game Explorer",
            icon="🎮",
        ),
        st.Page(
            "app/pages/model_performance.py",
            title="Model Performance",
            icon="📊",
        ),
        st.Page(
            "app/pages/methodology.py",
            title="Methodology & Limitations",
            icon="📘",
        ),
    ],
}


navigation = st.navigation(pages)
navigation.run()
