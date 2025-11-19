# 🧠 Smart Contract Generator (MCP + CrewAI + Streamlit)

An AI-powered system for **generating**, **validating**, and **interacting** with Solidity smart contracts using:

- MCP (Model Context Protocol) for tool execution  
- CrewAI for multi-step agent reasoning  
- Google Gemini for contract generation & reasoning  
- solcx + web3 + eth-tester for compilation & sandbox deployment  
- Streamlit for a conversational interface  

This system converts natural-language descriptions into complete Solidity contracts and validates them automatically.

---

## Features

- Generate Solidity contracts from text prompts  
- Compile contracts using solc  
- Deploy contracts on an in-memory Ethereum test chain  
- Built-in DuckDuckGo search tool  
- CrewAI multi-step agent pipeline  
- Interactive Streamlit chat interface  

---

## Project Structure

mcp_server/
solidity_generate_contract
solidity_check_deployability
web_search

agent.py
streamlit_app.py
mcp_tools.py
requirements.txt
README.md

yaml
Copy code

---

## MCP Tools (API Reference)

### 1. solidity_generate_contract

Generates Solidity code from a natural-language description.

**Inputs**
- `description` (str): Contract requirements  
- `blockchain` (str, optional): Target EVM chain (default: "Ethereum")

**Returns**
- SmartContract object containing:
  - contract_code  
  - clauses  
  - is_compilable = False  
  - is_deployable = False  

---

### 2. solidity_check_deployability

Compiles and deploys a Solidity contract using solcx and eth-tester.

**Inputs**
- `contract` (SmartContract)  
- `openzeppelin_path` (optional str)

**Returns**
- Updated SmartContract with:
  - is_compilable  
  - is_deployable  
  - compiler_errors  
  - deploy_errors  

---

### 3. web_search

Runs a DuckDuckGo search via DDGS.

**Inputs**
- `query` (str)  
- `max_results` (int)

**Returns**
- List of text snippets in the form "title\nbody\nhref"

---

## CrewAI Pipeline

Pipeline logic in `run_contract_pipeline()`.

### Agent
- Role: Blockchain Developer Agent  
- LLM: Google Gemini via Agentics  
- Memory enabled  
- 10 reasoning steps  

### Task 1 — Generate Contract
- Uses `solidity_generate_contract`  
- Saves result to memory key: `generated_contract`

### Task 2 — Validate Contract
- Uses `solidity_check_deployability`  
- Input: `{memory.generated_contract}`  

---

## Streamlit Chat Interface

Run:

streamlit run streamlit_app.py

yaml
Copy code

Features:
- Chat-based prompt entry  
- Display Solidity code  
- Compilation and deployment logs  
- Clause extraction  
- Multi-step iterative refinement  

---

## Running the MCP Server

Set environment variables in `.env`:

GEMINI_API_KEY=your_api_key
MCP_SERVER_PATH=./mcp_server.py

yaml
Copy code

Run the server:

python3 $MCP_SERVER_PATH

yaml
Copy code

CrewAI uses:

StdioServerParameters(
command="python3",
args=[os.getenv("MCP_SERVER_PATH")]
)

yaml
Copy code

---

## Installation

Install dependencies:

pip install -r requirements.txt

yaml
Copy code

---

## Requirements

agentics-python
python-dotenv
google-generativeai
crewai
crewai-tools
pyyaml
py-solc-x
web3
eth-tester
py-evm
streamlit
streamlit-elements
ddgs

yaml
Copy code

---

## Example

**User input:**
Create a token vesting contract with 12-month linear unlock.

yaml
Copy code

**Pipeline output:**
- Generated Solidity code  
- Clause list  
- is_compilable = True  
- is_deployable = True  

---

## Extending the Project

You can add new MCP tools:
- Security scanners  
- Gas estimators  
- Static analyzers  
- Contract upgrade advisors  

---

## Notes

- solcx may download compiler binaries on first run  
- eth-tester deployment is fully local  
- OpenZeppelin imports require `openzeppelin_path`