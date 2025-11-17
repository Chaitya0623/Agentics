import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from crewai import Agent, Task, Crew
from agentics import AG

from mcp_tools import SmartContract

load_dotenv()


def run_contract_pipeline(
    user_input: str, on_log: Optional[Callable[[str], None]] = None
) -> tuple[object, str]:
    fetch_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
        env={"UV_PYTHON": "3.12", **os.environ},
    )
    mcp_server_path = os.getenv("MCP_SERVER_PATH")
    if not mcp_server_path:
        raise RuntimeError(
            "MCP_SERVER_PATH is not set. Provide the path to your MCP server script in the environment."
        )

    mcp_params = StdioServerParameters(
        command="python3",
        args=[mcp_server_path],
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
            tools=[check_deployability],
            inputs={"contract": "{memory.generated_contract}"}
        )

        crew = Crew(
            agents=[contract_agent],
            tasks=[task_generate, task_validate],
            verbose=True,
        )

        log_buffer = io.StringIO()
        original_stdout = sys.stdout

        class _StdoutLogger:
            def write(self, data):
                original_stdout.write(data)
                log_buffer.write(data)
                if on_log:
                    on_log(log_buffer.getvalue())
                return len(data)

            def flush(self):
                original_stdout.flush()

        with redirect_stdout(_StdoutLogger()):
            result = crew.kickoff(inputs={"description": user_input})

        return result, log_buffer.getvalue()
