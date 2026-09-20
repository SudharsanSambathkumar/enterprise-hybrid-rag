"""Streamlit demo UI. Run: streamlit run app/streamlit_app.py"""

import streamlit as st

from hybrid_rag import HybridRAG

st.set_page_config(page_title="Enterprise Hybrid RAG", page_icon="🛒", layout="wide")


@st.cache_resource
def get_rag() -> HybridRAG:
    return HybridRAG().load()


EXAMPLES = {
    "📄 Vector: sick leave policy": "How many sick leaves do store staff get?",
    "🕸️ Graph: recall impact": "FreshDairy milk is recalled. Which stores and managers are affected?",
    "🤝 Hybrid: who + what to do": "Milk recall: what must each affected store manager do?",
    "🕸️ Graph: who supplies a store": "Which suppliers deliver products to the HSR Layout store?",
}
ROUTE_COLORS = {"vector": "#22d3ee", "graph": "#a78bfa", "hybrid": "#fbbf24"}

st.title("🛒 Enterprise Hybrid RAG")
st.caption("Vector RAG finds the right paragraph. Graph RAG connects the relationships. Hybrid does both.")

with st.sidebar:
    st.header("Settings")
    roles = st.multiselect("User roles", ["staff", "manager", "qa", "admin"], default=["manager"])
    route = st.radio("Route", ["auto", "vector", "graph", "hybrid"], horizontal=True)
    if st.button("Re-ingest data"):
        st.json(get_rag().ingest())
    st.divider()
    st.subheader("Try an example")
    for label, q in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["q"] = q

question = st.text_input("Ask a question", key="q", placeholder="e.g. Which stores stock BakeCo bread?")

if question:
    with st.spinner("Routing, retrieving and generating..."):
        ans = get_rag().query(question, roles=roles, route=None if route == "auto" else route)
    color = ROUTE_COLORS.get(ans.route, "#94a3b8")
    st.markdown(
        f"<span style='background:{color}22;border:1px solid {color};color:{color};"
        f"padding:3px 12px;border-radius:999px;font-weight:600'>Route: {ans.route.upper()}</span>"
        f"&nbsp; <span style='opacity:.7'>{ans.route_reason}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("### Answer")
    st.write(ans.answer)
    if ans.entities:
        st.caption("Graph entities: " + ", ".join(ans.entities))

    docs = [c for c in ans.contexts if c.kind == "doc"]
    paths = [c for c in ans.contexts if c.kind == "graph"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📄 Document evidence")
        for c in docs:
            with st.expander(f"[{c.ref}] {c.citation} · sim {c.raw_score:.2f}"):
                st.write(c.text)
        if not docs:
            st.caption("None for this route.")
    with col2:
        st.markdown("#### 🕸️ Graph paths")
        for c in paths:
            st.code(f"[{c.ref}] {c.text}", language=None)
        if not paths:
            st.caption("None for this route.")
