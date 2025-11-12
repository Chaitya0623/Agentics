import streamlit as st

# --- 1. Cached pipeline import ---
@st.cache_resource
def get_pipeline():
    from agent import run_contract_pipeline
    return run_contract_pipeline

# --- 2. Streamlit base config ---
st.set_page_config(page_title="🧠 Smart Contract Generator Chat", layout="wide")
st.title("🧠 Solidity Smart Contract Chat")

# --- 3. Initialize session state for chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 4. Sidebar instructions ---
st.sidebar.title("Instructions")
st.sidebar.write("""
- Chat with the AI to generate and refine Solidity contracts.
- Each message will generate Solidity code, validation results, and clauses.
- Continue the conversation to iteratively improve your contract.
""")

# --- 5. Display conversation in chatbot style ---
for i, message in enumerate(st.session_state.chat_history):
    if message["user"]:
        st.chat_message("user").write(message["user"])
    if message["contract"]:
        st.chat_message("assistant").write("Contract generated:")
        with st.chat_message("assistant"):
            # Solidity Code
            with st.expander("📜 Solidity Code", expanded=True):
                st.code(message["contract"].contract_code, language="solidity")
            
            # Validation Results
            with st.expander("🧪 Validation Results", expanded=True):
                validation_data = {
                    "is_compilable": getattr(message["contract"], "is_compilable", None),
                    "is_deployable": getattr(message["contract"], "is_deployable", None),
                    "compiler_errors": getattr(message["contract"], "compiler_errors", "None"),
                    "deploy_errors": getattr(message["contract"], "deploy_errors", "None"),
                }
                for k, v in validation_data.items():
                    st.write(f"• {k}: {v}")

            # Clauses
            with st.expander("📖 Clauses", expanded=True):
                clauses = getattr(message["contract"], "clauses", [])
                if clauses:
                    for clause in clauses:
                        st.write(f"• {clause.title} — {clause.description}")
                else:
                    st.write("No clauses found.")

# --- 6. Chat input for new messages ---
if user_input := st.chat_input("Describe or refine your contract..."):
    st.chat_message("user").write(user_input)

    pipeline = get_pipeline()
    with st.spinner("Generating response..."):
        # result, crew_log = pipeline(user_input)
        result = pipeline(user_input)

    # Extract contract data
    contract = None
    if hasattr(result, "tasks_output") and result.tasks_output: # how can we stream it
        last_task = result.tasks_output[-1]
        if hasattr(last_task, "pydantic") and last_task.pydantic:
            contract = last_task.pydantic
    elif hasattr(result, "pydantic") and result.pydantic:
        contract = result.pydantic
    elif hasattr(result, "raw") and result.raw:
        contract = result.raw

    # Append to chat history
    st.session_state.chat_history.append({
        "user": user_input,
        "contract": contract
    })

    # Display AI response immediately
    with st.chat_message("assistant"):
        if contract:
            for task in result.tasks_output:
                st.info(f"Task executed")    
                with st.expander("Description"):
                    st.write(getattr(task, "description", "No description available."))

            st.success("Agent has completed all tasks successfully! The contract is ready.")
            with st.expander("📜 Solidity Code", expanded=True):
                st.code(contract.contract_code, language="solidity")
            with st.expander("🧪 Validation Results", expanded=True):
                validation_data = {
                    "is_compilable": getattr(contract, "is_compilable", None),
                    "is_deployable": getattr(contract, "is_deployable", None),
                    "compiler_errors": getattr(contract, "compiler_errors", "None"),
                    "deploy_errors": getattr(contract, "deploy_errors", "None"),
                }
                for k, v in validation_data.items():
                    st.write(f"• {k}: {v}")
            with st.expander("📖 Clauses", expanded=True):
                clauses = getattr(contract, "clauses", [])
                if clauses:
                    for clause in clauses:
                        st.write(f"• {clause.title} — {clause.description}")
                else:
                    st.write("No clauses found.")
        else:
            st.error("No contract generated.")
        
        # st.write(crew_log)
