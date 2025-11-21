import re
import streamlit as st
import streamlit.components.v1 as components
import html as _html

# ---------------------------
# Configuration
# ---------------------------
LOG_HEIGHT_PX = 300  # adjust log box height here

# ---------------------------
# 1. LOAD PIPELINE (cached)
# ---------------------------
@st.cache_resource
def get_pipeline():
    from agent import run_contract_pipeline
    return run_contract_pipeline

# ---------------------------
# 2. PAGE CONFIG + CSS
# ---------------------------
st.set_page_config(
    page_title="Solidity Smart Contract Generator",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #fafafa;
}
.sidebar .sidebar-content {
    background-color: #1a1d23;
}
button[kind="primary"] {
    background-color: #4b8bf4 !important;
    color: white !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input {
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 3. INTRO SCREEN
# ---------------------------
if "show_instructions" not in st.session_state:
    st.session_state.show_instructions = True

if st.session_state.show_instructions:
    st.title("👋 Welcome to Solidity Smart Contract Generator")
    st.markdown("""
    **How to use:**
    1. Enter your contract idea  
    2. Click **Generate Contract**  
    3. See logs + final output together  
    """)
    if st.button("Got it!"):
        st.session_state.show_instructions = False

    # ---------------------------
    # Credits block
    # ---------------------------
    st.markdown(
        """
    <div style="margin-top: 1.5rem; font-size: 0.9rem; color: #cccccc; line-height: 1.5;">

    <strong>Project Team</strong><br/>
    <a href="https://www.linkedin.com/in/chaityas/" target="_blank">Chaitya Shah</a><br/>
    <a href="https://www.linkedin.com/in/chunghyun-han-355b80244/" target="_blank">Chunghyun Han</a><br/>
    <a href="https://www.linkedin.com/in/nami-jain/" target="_blank">Nami Jain</a><br/>
    <a href="https://www.linkedin.com/in/yegan-dhaivakumar" target="_blank">Yegan Dhaivakumar</a><br/><br/>

    <strong>Faculty Advisors</strong><br/>
    <a href="https://www.linkedin.com/in/gliozzo/" target="_blank">Alfio Massimiliano Gliozzo</a><br/>
    <a href="https://www.linkedin.com/in/agostino-capponi-842b41a5/" target="_blank">Agostino Capponi</a>

    </div>
    """,
        unsafe_allow_html=True
    )
    st.stop()

# ---------------------------
# 4. SESSION STATE
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "crew_log" not in st.session_state:
    st.session_state.crew_log = ""

_ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
def clean_ansi(text: str):
    return _ansi_escape.sub("", text or "")

# ---------------------------
# 5. LOG RENDERER (single placeholder, forced autoscroll)
# ---------------------------
def render_logs_in_placeholder(placeholder, log_text: str, height_px: int = LOG_HEIGHT_PX):
    """Render logs inside a single placeholder using components.html(), autoscrolls to bottom."""
    safe_text = _html.escape(log_text).replace("\n", "<br/>")
    html = f"""
    <div id="log-box" style="
        background-color: #1a1d23;
        color: #fafafa;
        padding: 12px;
        border-radius: 6px;
        height: {height_px}px;
        overflow-y: auto;
        font-family: monospace;
        border: 1px solid #333;
        white-space: pre-wrap;
        overflow-anchor: none;
    ">{safe_text}</div>

    <script>
    setTimeout(function() {{
        var box = document.getElementById("log-box");
        if (box) {{
            box.scrollTop = box.scrollHeight;
        }}
    }}, 20);
    </script>
    """
    placeholder.empty()
    with placeholder:
        components.html(html, height=height_px + 30, scrolling=False)

# ---------------------------
# 6. SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("💬 Chat Assistant / Model")

    example_prompts = [
        "Create an ERC20 token with minting and burning",
        "Write an NFT contract with royalties",
        "Build a DAO voting contract",
        "Add a time lock to a smart contract",
    ]
    for prompt in example_prompts:
        if st.button(prompt):
            st.session_state["chat_input"] = prompt

    MODEL_OPTIONS = {
        "Gemini 2.0 Flash": "gemini",
        "FSM Fine-Tuned TinyLlama": "fsm_pretrained",
        "GPT-4": "gpt4",
        "Claude 3.5": "claude_3_5",
    }

    model_label = st.selectbox("Choose generation model", list(MODEL_OPTIONS.keys()))
    selected_model = MODEL_OPTIONS[model_label]

    user_input = st.text_area("Ask or refine your contract...", key="chat_input")

# ---------------------------
# 7. MAIN APP
# ---------------------------
st.title("🧠 Solidity Contract Generator")

# Create one persistent placeholder for logs (shared across updates)
log_placeholder = st.empty()

if st.button("🚀 Generate Contract"):
    if not user_input.strip():
        st.warning("Please enter a contract description.")
        st.stop()

    # show user message in chat area
    st.chat_message("user").write(user_input)

    # reset logs
    st.session_state.crew_log = ""
    render_logs_in_placeholder(log_placeholder, "Waiting for logs...", LOG_HEIGHT_PX)

    pipeline = get_pipeline()

    # define log callback
    def _handle_log_stream(log_text: str):
        st.session_state.crew_log = clean_ansi(log_text)
        render_logs_in_placeholder(log_placeholder, st.session_state.crew_log, LOG_HEIGHT_PX)

    with st.chat_message("assistant"):
        st.write("🛠 Running agents… generating contract...")

        # run pipeline
        with st.spinner("Generating contract..."):
            result, final_log = pipeline(
                user_input,
                model_choice=selected_model,
                on_log=_handle_log_stream,
            )

        # render final log
        st.session_state.crew_log = clean_ansi(final_log or st.session_state.crew_log)
        render_logs_in_placeholder(log_placeholder, st.session_state.crew_log, LOG_HEIGHT_PX)

        # ---------------------------
        # EXTRACT CONTRACT
        # ---------------------------
        contract = None
        if hasattr(result, "tasks_output") and result.tasks_output:
            last_task = result.tasks_output[-1]
            if hasattr(last_task, "pydantic") and last_task.pydantic:
                contract = last_task.pydantic
        elif hasattr(result, "pydantic") and result.pydantic:
            contract = result.pydantic
        elif hasattr(result, "raw") and result.raw:
            contract = result.raw

        st.session_state.chat_history.append({"user": user_input, "contract": contract})

        # ---------------------------
        # FINAL OUTPUT
        # ---------------------------
        st.subheader("✅ Final Generated Contract")

        if contract and hasattr(contract, "contract_code"):
            st.code(contract.contract_code, language="solidity")
        else:
            st.write("⚠️ No contract code returned.")

        st.subheader("🧪 Validation Results")
        validation_data = {
            "is_compilable": getattr(contract, "is_compilable", None),
            "is_deployable": getattr(contract, "is_deployable", None),
            "compiler_errors": getattr(contract, "compiler_errors", "None"),
            "deploy_errors": getattr(contract, "deploy_errors", "None"),
        }
        for k, v in validation_data.items():
            st.write(f"• **{k}:** {v}")

        st.subheader("📖 Clauses")
        clauses = getattr(contract, "clauses", [])
        if clauses:
            for clause in clauses:
                title = getattr(clause, "title", clause.get("title") if isinstance(clause, dict) else None)
                desc = getattr(clause, "description", clause.get("description") if isinstance(clause, dict) else None)
                st.write(f"• **{title}** — {desc}")
        else:
            st.write("No clauses found.")
