# streamlit_app.py
import re

import streamlit as st

# --- Page Setup ---
st.set_page_config(
    page_title="Solidity Smart Contract Generator",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS for professional look ---
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

# --- Temporary Instructions Screen ---
if "show_instructions" not in st.session_state:
    st.session_state.show_instructions = True

if st.session_state.show_instructions:
    st.title("👋 Welcome to Solidity Smart Contract Generator")
    st.markdown("""
    **How to use:**
    1. Choose a model from the dropdown.  
    2. Enter your contract idea in natural language.  
    3. Click **Generate Contract** to see the Solidity code.  
    """)
    if st.button("Got it!"):
        st.session_state.show_instructions = False
    st.stop()

# --- Sidebar: Chat Assistant + Example Prompts ---
with st.sidebar:
    st.header("💬 Chat Assistant")
    st.markdown("**Try asking:**")

    example_prompts = [
        "Create an ERC20 token with minting and burning",
        "Write an NFT contract with royalties",
        "Build a DAO voting contract",
        "Add a time lock to a smart contract"
    ]
    for prompt in example_prompts:
        if st.button(prompt):
            st.session_state["chat_input"] = prompt

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.markdown(f"**You:** {msg['user']}")
        st.markdown(f"**Bot:** {msg['bot']}")

    user_input = st.text_input("Ask about smart contracts...", key="chat_input")
    if st.button("Send"):
        if user_input:
            bot_reply = f"Placeholder reply for: '{user_input}'"
            st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})
            st.session_state.chat_input = ""
            st.experimental_rerun()

# --- Main Tabs ---
st.title("🧠 Solidity Smart Contract Generator")
st.markdown("Describe your contract in natural language and generate Solidity code instantly.")

tabs = st.tabs(["⚙️ Generator", "📜 CrewAI Logs", "💾 Deploy / Test"])

with tabs[0]:
    st.subheader("Contract Generator")
    model = st.selectbox("Choose generation model", ["Gemini 2.0 Flash", "GPT-4", "Claude 3.5"])
    user_prompt = st.text_area("Describe your smart contract idea...")
    if st.button("🚀 Generate Contract"):
        st.code("// Solidity code will appear here", language="solidity")

with tabs[1]:
    st.subheader("CrewAI Logs")
    st.info("Crew logs will appear here once a run starts.")

with tabs[2]:
    st.subheader("Deploy / Test")
    st.markdown("Future feature: simulate or deploy contracts directly.")