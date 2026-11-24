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
st.caption("შენი პირადი კონტენტ მენეჯერი: იდეები, კალენდარი, სცენარები და ტრენდები ერთ ადგილას.")


@st.cache_resource
def get_graph():
    """გრაფი ერთხელ აიგება და ყველა სესიისთვის გამოიყენება (საუბრები thread_id-ით გამოიყოფა)."""
    return build_graph()


try:
    graph = get_graph()
except Exception as exc:
    st.error("🔑 აგენტი ვერ გაეშვა: GROQ_API_KEY არ არის მითითებული ან არასწორია. "
             "დაამატე Streamlit Secrets-ში.")
    st.caption(f"დეტალი: {exc}")
    st.stop()


def new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []       # ეკრანზე გამოსაჩენი შეტყობინებები
    st.session_state.pending = None  # დადასტურების მოლოდინში მყოფი მოქმედება


if "thread_id" not in st.session_state:
    new_conversation()

with st.sidebar:
    if st.button("🆕 ახალი საუბარი"):
        new_conversation()
        st.rerun()

config = {"configurable": {"thread_id": st.session_state.thread_id}}

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


def handle(out):
    """პასუხის დამუშავება: ტექსტი, დადასტურების მოთხოვნა ან შეცდომა."""
    if out["type"] == "interrupt":
        st.session_state.pending = out["payload"]
    elif out["type"] == "error":
        # pending უცვლელი რჩება, რომ დადასტურების ღილაკებით ისევ სცადოს
        st.session_state.chat.append({"role": "assistant", "content": out["text"]})
    else:
        st.session_state.pending = None
        st.session_state.chat.append({"role": "assistant", "content": out["text"]})


# დადასტურების ბარათი (წაშლამდე)
if st.session_state.pending:
    st.warning("⚠️ " + st.session_state.pending["message"])
    col1, col2 = st.columns(2)
    if col1.button("✅ დიახ, წაშალე"):
        with st.spinner("..."):
            handle(run_turn(graph, config, resume=True))
        st.rerun()
    if col2.button("❌ არა, გაუქმება"):
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
