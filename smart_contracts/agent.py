import os
from dotenv import load_dotenv

from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from crewai import Agent, Task, Crew
from agentics import AG

from mcp_tools import SmartContract
# import io
# import sys

# buffer = io.StringIO()
# sys_stdout = sys.stdout
# sys.stdout = buffer

load_dotenv()

def run_contract_pipeline(user_input: str) -> SmartContract:
    fetch_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
        env={"UV_PYTHON": "3.12", **os.environ},
    )
    mcp_params = StdioServerParameters(
        command="python3",
        args=[os.getenv("MCP_SERVER_PATH")],
        env={"UV_PYTHON": "3.12", **os.environ},
    ) # check if you should mention these parameters outside the pipeline

    with MCPServerAdapter(fetch_params) as fetch_tools, MCPServerAdapter(
        mcp_params
    ) as mcp_tools:
        # print(f"Available tools from Stdio MCP server: {[tool.name for tool in fetch_tools]}")
        # print(f"Available tools from Stdio MCP server: {[tool.name for tool in mcp_tools]}")

        for t in mcp_tools:
            globals()[t.name] = t

        tools = fetch_tools + mcp_tools

        contract_agent = Agent(
            role="Blockchain Developer Agent",
            goal="Generate Solidity smart contracts from natural language descriptions.",
            backstory="An expert Solidity developer specializing in transforming user requirements into blockchain-ready smart contracts.",
            tools=tools,
            reasoning=True,
            reasoning_steps=10,
            memory=True,
            verbose=True,
            llm=AG.get_llm_provider('gemini')
        )

        task_generate = Task(
            description="Generate a Solidity smart contract from the given natural language description.",
            expected_output="A SmartContract object containing Solidity code and contract clauses.",
            agent=contract_agent,
            output_pydantic=SmartContract,
            tools=[generate_smart_contract],
            memory_key="generated_contract"
        )

        task_validate = Task(
            description="Check if the generated Solidity smart contract can compile and deploy successfully.",
            expected_output="An updated SmartContract object with compilation/deployment status and error logs.",
            agent=contract_agent,
            output_pydantic=SmartContract,
            tools=[validate_smart_contract],
            inputs={"contract": "{memory.generated_contract}"}
        )

        # Crew Orchestration
        crew = Crew(
            agents=[contract_agent],
            tasks=[task_generate, task_validate], # make other validation tests not run sequentially
            verbose=True,
        )

        # user_input = input("Enter a natural language description of the contract: ").strip()
        result = crew.kickoff(inputs={"description": user_input})

        # sys.stdout = sys_stdout
        # crew_log = buffer.getvalue()
        # return result, crew_log
        return result