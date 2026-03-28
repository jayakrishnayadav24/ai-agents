import streamlit as st
import uuid
import os
import re
from main import run_orchestrator_streaming
from agents.memory import ensure_table_exists, save_display, load_display_history, list_sessions

st.set_page_config(page_title="AWS Multi-Agent", page_icon="🤖", layout="wide")

def clean_text(text: str) -> str:
    """Strip XML tags that Nova Lite leaks into responses."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<response>(.*?)</response>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<end_turn>", "", text)
    return text.strip()

# ── Init ──────────────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}   # {session_id: [{role, content}]}

try:
    ensure_table_exists()
except Exception:
    pass

def get_chat(sid):
    return st.session_state.chat_history.get(sid, [])

def add_chat(sid, role, content):
    st.session_state.chat_history.setdefault(sid, []).append({"role": role, "content": content})

def switch_session(sid):
    st.session_state.session_id = sid
    # Load from DynamoDB if not already in memory
    if sid not in st.session_state.chat_history:
        history = load_display_history(sid)
        st.session_state.chat_history[sid] = [
            {"role": m["role"], "content": m["text"]} for m in history
        ]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 AWS Multi-Agent")

    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        new_sid = str(uuid.uuid4())[:8]
        switch_session(new_sid)
        st.rerun()

    st.divider()

    # Session list — each is a clickable button
    st.markdown("**💬 Chats**")
    try:
        sessions = list_sessions()
        # Always show current session even if it has no messages yet
        current = st.session_state.session_id
        existing_ids = [s["session_id"] for s in sessions]
        if current not in existing_ids:
            sessions = [{"session_id": current, "title": "New chat"}] + sessions

        for s in sessions:
            sid   = s["session_id"]
            title = s["title"] if len(s["title"]) <= 35 else s["title"][:35] + "…"
            is_active = sid == st.session_state.session_id
            label = f"**› {title}**" if is_active else title
            if st.button(label, key=f"sess_{sid}", use_container_width=True):
                switch_session(sid)
                st.rerun()
    except Exception as e:
        st.caption(f"Could not load sessions: {e}")

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id}`")
    st.caption(f"Region: `{os.getenv('AWS_REGION', 'ap-south-1')}`")
    st.caption(f"Profile: `{os.getenv('AWS_PROFILE', 'default')}`")

# ── Main chat area ────────────────────────────────────────────────────────────
sid = st.session_state.session_id
st.title("🤖 AWS Multi-Agent System")
st.caption("Powered by Amazon Bedrock · Type a natural language request below")

# Render current session messages
for msg in get_chat(sid):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("e.g. Diagnose EC2 instance i-0abc123, Create ECS cluster prod...")

if prompt:
    add_chat(sid, "user", prompt)
    save_display(sid, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box   = st.status("🔄 Agents working...", expanded=True)
        result_lines = []
        final        = None

        for event in run_orchestrator_streaming(
            prompt,
            region=os.getenv("AWS_REGION", "ap-south-1"),
            session_id=sid,
        ):
            kind = event["kind"]

            if kind == "agent_start":
                status_box.write(f"🚀 **{event['agent'].upper()}** → {event['task']}")

            elif kind == "tool_call":
                inputs_str = ", ".join(f"{k}={v}" for k, v in event["inputs"].items())
                status_box.write(f"　🔧 `{event['tool']}` ({inputs_str})")

            elif kind == "tool_result":
                icon = "✅" if event["status"] != "error" else "❌"
                status_box.write(f"　{icon} {event['message']}")

            elif kind == "agent_done":
                icon = "✅" if event["status"] == "success" else "⚠️"
                result_lines.append(f"{icon} **{event['agent'].upper()}**: {event['message']}")

            elif kind == "summary":
                status_box.update(label="✅ Done", state="complete", expanded=False)
                final = event["text"]

        # Fix: if no summary event, build answer from agent_done lines
        answer = clean_text(final or "\n\n".join(result_lines) or "No response from agents.")
        st.markdown(answer)
        add_chat(sid, "assistant", answer)
        save_display(sid, "assistant", answer)
        st.rerun()
