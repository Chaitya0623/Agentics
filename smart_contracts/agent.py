import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai

from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from crewai import Agent, Task, Crew
import yaml
from agentics import AG

# ========================
# Load environment & Gemini setup
# ========================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ========================
# Pydantic Schemas
# ========================
class ContractClause(BaseModel):
    title: str
    description: str

class SmartContract(BaseModel):
    contract_code: str = Field(..., description="The Solidity smart contract code")
    clauses: List[ContractClause]

# ========================
# Smart Contract Generation Function
# ========================
def generate_smart_contract(description: str, blockchain: str = "Ethereum") -> SmartContract:
    prompt = f"""
    You are an expert blockchain developer.
    Generate a complete Solidity smart contract for the following description:
    "{description}"

    Requirements:
    - Contract must be valid for {blockchain}.
    - Include functions, modifiers, and comments.
    - Provide only the Solidity code and a list of key clauses (title + description).
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Return structured result
    return SmartContract(
        contract_code=response.text,
        clauses=[
            ContractClause(title="Example Clause", description="Core logic or rule implemented here.")
        ],
    )

# Dummy function retained for structure
def get_ticker_from_name(input_str: str) -> str:
    return "N/A"

# ========================
# CrewAI + MCP Integration
# ========================
fetch_params = StdioServerParameters(
    command="uvx",
    args=["mcp-server-fetch"],
    env={"UV_PYTHON": "3.12", **os.environ},
)
search_params = StdioServerParameters(
    command="python3",
    args=[os.getenv("MCP_SERVER_PATH")],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with MCPServerAdapter(fetch_params) as fetch_tools, MCPServerAdapter(
    search_params
) as search_tools:
    print(f"Available tools from Stdio MCP server: {[tool.name for tool in fetch_tools]}")
    print(f"Available tools from Stdio MCP server: {[tool.name for tool in search_tools]}")

    tools = fetch_tools + search_tools

    # Define Smart Contract Generation Agent
    contract_agent = Agent(
        role="Blockchain Developer",
        goal="Generate Solidity smart contracts from natural language descriptions.",
        backstory="An expert Solidity developer specializing in transforming user requirements into blockchain-ready smart contracts.",
        tools=tools,
        reasoning=True,
        reasoning_steps=10,
        memory=True,
        verbose=True,
        llm=AG.get_llm_provider('gemini')
    )

    # Define Task
    contract_task = Task(
        description="""Generate a Solidity smart contract from the given natural language specification.""",
        expected_output="""A Solidity smart contract and a list of clauses.""",
        agent=contract_agent,
        output_pydantic=SmartContract,
    )

    # Crew Orchestration
    crew = Crew(
        agents=[contract_agent],
        tasks=[contract_task],
        verbose=True,
    )

    # ========================
    # Direct Execution (no conversation loop)
    # ========================
    user_input = input("Enter a natural language description of the contract: ").strip()
    result = crew.kickoff(inputs={"description": user_input})

    if result.pydantic:
        ai_output = yaml.dump(result.pydantic.model_dump(), sort_keys=False)
    else:
        ai_output = str(result)

    print("AI:", ai_output)