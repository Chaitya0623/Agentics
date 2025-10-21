# streamlit_app.py
import streamlit as st

@st.cache_resource
def get_pipeline():
    from agent import run_contract_pipeline
    return run_contract_pipeline

st.set_page_config(page_title="Smart Contract Generator", layout="wide")

st.title("🧠 Solidity Smart Contract Generator")
st.write("Describe your contract in natural language and generate Solidity code instantly.")

# user_input = st.text_area(
#     "Enter contract description:",
#     placeholder=""""Create an ERC20 token named "MyToken" with symbol "MTK" and a total supply of 1,000,000 tokens (18 decimals). The contract should include standard ERC20 functions: transfer, approve, transferFrom, balanceOf, and totalSupply, as well as mint (only by owner) and burn (any holder) functions. Include events for Transfer and Approval, proper access control, and input validation following Solidity best practices.""",
#     height=200
# )
user_input = """Create an ERC20 token named "MyToken" with symbol "MTK" and a total supply of 1,000,000 tokens (18 decimals). The contract should include standard ERC20 functions: transfer, approve, transferFrom, balanceOf, and totalSupply, as well as mint (only by owner) and burn (any holder) functions. Include events for Transfer and Approval, proper access control, and input validation following Solidity best practices."""

if st.button("🚀 Generate Contract"):
    if not user_input.strip():
        st.warning("Please enter a contract description.")
    else:
        with st.spinner("Generating and validating your contract..."):
            pipeline = get_pipeline()
            result = pipeline(user_input)
        st.success("✅ Contract generated successfully!")

        # st.write(result)
        contract = None

        if hasattr(result, "tasks_output") and result.tasks_output:
            # Get last task output (usually validation)
            last_task = result.tasks_output[-1]  # TaskOutput object
            if hasattr(last_task, "pydantic") and last_task.pydantic:
                contract = last_task.pydantic
        elif hasattr(result, "pydantic") and result.pydantic:
            contract = result.pydantic
        elif hasattr(result, "raw") and result.raw:
            contract = result.raw

        # --- Display in Streamlit ---
        if contract and hasattr(contract, "contract_code"):
            st.subheader("📜 Solidity Code")
            st.code(contract.contract_code, language="solidity")

            st.subheader("🧪 Validation Results")
            st.json({
                "is_compilable": contract.is_compilable,
                "is_deployable": contract.is_deployable,
                "compiler_errors": contract.compiler_errors or "None",
                "deploy_errors": contract.deploy_errors or "None"
            })

            if hasattr(contract, "clauses") and contract.clauses:
                with st.expander("📖 Contract Clauses"):
                    for clause in contract.clauses:
                        st.markdown(f"**{clause.title}** — {clause.description}")
        else:
            st.error("❌ No valid SmartContract output found. Check your Crew pipeline or task names.")