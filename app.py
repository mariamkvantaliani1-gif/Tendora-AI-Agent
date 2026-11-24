"""Trendora AI: Streamlit ჩატის გვერდი. გაშვება: streamlit run app.py"""
import os
import uuid

import streamlit as st

# გასაღები: Streamlit Secrets-იდან ან გარემოს ცვლადიდან (კოდში არასდროს ვწერთ)
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from agent import build_graph, run_turn  # noqa: E402

st.set_page_config(page_title="Trendora AI", page_icon="📈")
st.title("📈 Trendora AI")
st.caption("შენი პირადი კონტენტ მენეჯერი")


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []      # ეკრანზე გამოსაჩენი შეტყობინებები
    st.session_state.pending = None  # დადასტურების მოლოდინში მყოფი მოქმედება

config = {"configurable": {"thread_id": st.session_state.thread_id}}

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


def handle(out):
    if out["type"] == "interrupt":
        st.session_state.pending = out["payload"]
    else:
        st.session_state.pending = None
        st.session_state.chat.append({"role": "assistant", "content": out["text"]})


# დადასტურების ბარათი (წაშლამდე)
if st.session_state.pending:
    st.warning("⚠️ " + st.session_state.pending["message"])
    col1, col2 = st.columns(2)
    if col1.button("დიახ, წაშალე"):
        with st.spinner("..."):
            handle(run_turn(graph, config, resume=True))
        st.rerun()
    if col2.button("არა, გაუქმება"):
        with st.spinner("..."):
            handle(run_turn(graph, config, resume=False))
        st.rerun()
else:
    prompt = st.chat_input("დაწერე შეტყობინება...")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.spinner("Trendora ფიქრობს..."):
            handle(run_turn(graph, config, user_text=prompt))
        st.rerun()
