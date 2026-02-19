import time
import random
import io
from datetime import datetime
import streamlit as st
import os

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from pypdf import PdfReader


# -----------------------------
# Config
# -----------------------------

ENDPOINT = "https://legalai-agent2.services.ai.azure.com/api/projects/legalai-agent2-project"
AGENT_NAME = "Legal-Agent2"

MIN_SECONDS_BETWEEN_REQUESTS = 2.5
MAX_RETRIES_429 = 6

# ✅ token control
MAX_DOC_CHARS = 12000
MAX_PDF_PAGES = 40

st.set_page_config(page_title="Legal Agent (Foundry)", page_icon="⚖️", layout="centered")
st.title("⚖️ Legal Agent")
st.caption("Token-optimized • Streaming • Conversation memory")


# -----------------------------
# Clients
# -----------------------------
@st.cache_resource
def get_clients():
    os.environ.setdefault("OPENAI_API_VERSION", "2024-12-01-preview")

    project = AIProjectClient(
        endpoint=ENDPOINT,
        credential=DefaultAzureCredential(),
    )

    return project, project.agents, project.get_openai_client()


project, agents_ops, openai_client = get_clients()


# -----------------------------
# Agent lookup
# -----------------------------
def get_agent_by_name(agents_ops, name):
    agent = next((a for a in agents_ops.list() if getattr(a, "name", None) == name), None)
    if not agent:
        raise ValueError("Agent not found")
    return agent


# -----------------------------
# Rate limit helpers
# -----------------------------
def is_rate_limit_error(e):
    return "429" in str(e) or "rate" in str(e).lower()


def backoff_sleep(attempt):
    delay = min(20, 0.8 * (2 ** attempt))
    delay *= (0.75 + random.random() * 0.5)
    time.sleep(delay)


def throttle_guard():
    now = datetime.utcnow()
    last = st.session_state.get("last_request")
    if last:
        diff = (now - last).total_seconds()
        if diff < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - diff)
    st.session_state["last_request"] = datetime.utcnow()


# -----------------------------
# File extraction
# -----------------------------
def extract_text(uploaded):
    name = uploaded.name.lower()
    raw = uploaded.getvalue()

    if name.endswith((".txt", ".md")):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        pages = reader.pages[:MAX_PDF_PAGES]
        return "\n".join(p.extract_text() or "" for p in pages)

    return ""


def truncate_text(text):
    if len(text) > MAX_DOC_CHARS:
        return text[:MAX_DOC_CHARS] + "\n[TRUNCATED]"
    return text


# -----------------------------
# Streaming call
# -----------------------------
def stream_agent(agent_name, conv_id, text):

    if st.session_state.get("in_flight"):
        st.warning("Request already running")
        return

    st.session_state.in_flight = True

    try:
        throttle_guard()

        for attempt in range(MAX_RETRIES_429 + 1):
            try:
                stream = openai_client.responses.create(
                    conversation=conv_id,
                    input=text,
                    stream=True,
                    max_output_tokens=1000,  # ✅ reduced
                    extra_body={
                        "agent": {
                            "type": "agent_reference",
                            "name": agent_name,
                        }
                    },
                )

                for event in stream:
                    if getattr(event, "type", None) == "response.output_text.delta":
                        yield event.delta
                return

            except Exception as e:
                if is_rate_limit_error(e) and attempt < MAX_RETRIES_429:
                    backoff_sleep(attempt)
                    continue
                raise

    finally:
        st.session_state.in_flight = False


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = get_agent_by_name(agents_ops, AGENT_NAME)

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = openai_client.conversations.create(items=[]).id

if "doc_sent" not in st.session_state:
    st.session_state.doc_sent = False

agent = st.session_state.agent


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.success(f"Agent: {agent.name}")
st.sidebar.write(f"Conversation: {st.session_state.conversation_id}")

uploaded = st.sidebar.file_uploader("Upload TXT/MD/PDF", type=["txt", "md", "pdf"])

if uploaded:
    text = truncate_text(extract_text(uploaded))

    if text:
        st.sidebar.success("Text extracted")

        # ✅ store once only
        if not st.session_state.doc_sent:
            openai_client.responses.create(
                conversation=st.session_state.conversation_id,
                input=f"Reference document:\n{text}",
                max_output_tokens=200,
                extra_body={
                    "agent": {
                        "type": "agent_reference",
                        "name": agent.name,
                    }
                },
            )
            st.session_state.doc_sent = True
            st.sidebar.info("Document stored once in memory")
    else:
        st.sidebar.warning("No readable text found")


if st.sidebar.button("🧹 Clear chat"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🆕 New conversation"):
    st.session_state.conversation_id = openai_client.conversations.create(items=[]).id
    st.session_state.messages = []
    st.session_state.doc_sent = False
    st.rerun()


# -----------------------------
# Render history
# -----------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Ask…")

if prompt:

    prompt = prompt[:1500]  # ✅ guard length
    user_input = prompt + "\nAnswer concisely."

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        box = st.empty()
        full = ""

        try:
            for chunk in stream_agent(agent.name, st.session_state.conversation_id, user_input):
                full += chunk
                box.markdown(full)
        except Exception as e:
            full = f"Error: {e}"
            box.error(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
