import io
import os
import sys
import types
import warnings
from contextlib import redirect_stdout
from importlib import util
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


def _ensure_crewai_rag_shim() -> None:
    """Provide a lightweight shim for crewai.rag if the optional extras are missing."""
    if util.find_spec("crewai.rag") is not None:
        return

    if "crewai.rag" in sys.modules:
        return

    warnings.warn(
        "crewai.rag extras are not installed. Using a minimal compatibility shim. "
        "Install CrewAI with its RAG extras for full functionality.",
        RuntimeWarning,
        stacklevel=2,
    )

    rag_pkg = types.ModuleType("crewai.rag")
    rag_pkg.__path__ = []  # mark as package
    sys.modules["crewai.rag"] = rag_pkg

    embeddings_mod = types.ModuleType("crewai.rag.embeddings")
    sys.modules["crewai.rag.embeddings"] = embeddings_mod

    factory_mod = types.ModuleType("crewai.rag.embeddings.factory")

    def _missing_embedding_function(*args, **kwargs):
        raise ImportError(
            "crewai.rag embeddings are unavailable. Install crewai with RAG extras."
        )

    factory_mod.get_embedding_function = _missing_embedding_function
    sys.modules["crewai.rag.embeddings.factory"] = factory_mod

    class _BaseConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    chromadb_mod = types.ModuleType("crewai.rag.chromadb")
    chromadb_config_mod = types.ModuleType("crewai.rag.chromadb.config")

    class ChromaDBConfig(_BaseConfig):
        pass

    chromadb_config_mod.ChromaDBConfig = ChromaDBConfig
    chromadb_mod.config = chromadb_config_mod
    sys.modules["crewai.rag.chromadb"] = chromadb_mod
    sys.modules["crewai.rag.chromadb.config"] = chromadb_config_mod

    qdrant_mod = types.ModuleType("crewai.rag.qdrant")
    qdrant_config_mod = types.ModuleType("crewai.rag.qdrant.config")

    class QdrantConfig(_BaseConfig):
        pass

    qdrant_config_mod.QdrantConfig = QdrantConfig
    qdrant_mod.config = qdrant_config_mod
    sys.modules["crewai.rag.qdrant"] = qdrant_mod
    sys.modules["crewai.rag.qdrant.config"] = qdrant_config_mod

    rag_pkg.embeddings = embeddings_mod
    rag_pkg.chromadb = chromadb_mod
    rag_pkg.qdrant = qdrant_mod


_ensure_crewai_rag_shim()

from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from crewai import Agent, Task, Crew
from agentics import AG

from mcp_tools import SmartContract

GENERATION_TOOL_MAP = {
    "gemini": "generate_smart_contract",
    "fsm_pretrained": "generate_smart_contract_pretrained",
}

load_dotenv()


def run_contract_pipeline(
    user_input: str,
    model_choice: str = "gemini",
    on_log: Optional[Callable[[str], None]] = None,
) -> tuple[object, str]:
    fetch_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
        env={"UV_PYTHON": "3.12", **os.environ},
    )
    mcp_server_path = os.getenv("MCP_SERVER_PATH")
    default_candidates = [
        PROJECT_ROOT / "smart_contracts" / "mcp_tools.py",
        PROJECT_ROOT / "mcp" / "DDG_search_tool_mcp.py",
    ]

    def _resolve_default():
        for candidate in default_candidates:
            if candidate.exists():
                return str(candidate)
        return None

    resolved_path = Path(mcp_server_path).expanduser() if mcp_server_path else None
    if not resolved_path or not resolved_path.exists():
        fallback = _resolve_default()
        if fallback:
            mcp_server_path = fallback
        else:
            raise RuntimeError(
                (
                    f"MCP server script not found at {mcp_server_path}"
                    if mcp_server_path
                    else "MCP_SERVER_PATH is not set"
                )
                + ". Provide MCP_SERVER_PATH or ensure a default MCP server script exists."
            )
    else:
        mcp_server_path = str(resolved_path)

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

        tool_registry = {t.name: t for t in mcp_tools}
        for name, tool in tool_registry.items():
            globals()[name] = tool

        tools = fetch_tools + list(tool_registry.values())

        generation_key = (model_choice or "gemini").lower()
        generation_tool_name = GENERATION_TOOL_MAP.get(
            generation_key, GENERATION_TOOL_MAP["gemini"]
        )
        generation_tool = tool_registry.get(generation_tool_name)
        if generation_tool is None:
            raise RuntimeError(
                f"Requested generation tool '{generation_tool_name}' is not available from the MCP server."
            )

        check_tool = tool_registry.get("validate_smart_contract")
        if check_tool is None:
            raise RuntimeError("validate_smart_contract tool is not available.")

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
            tools=[generation_tool],
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
