import os
import sys
from pathlib import Path
from threading import Lock
from typing import List, Optional

from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import google.generativeai as genai
import solcx
from web3 import Web3
from eth_tester import EthereumTester
from eth_tester.exceptions import TransactionFailed
from ddgs import DDGS

mcp = FastMCP("Search")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DEFAULT_FSM_CHECKPOINT = "checkpoint-1200"
FSM_ADAPTER_ROOT = (
    PROJECT_ROOT
    / "smart_contracts"
    / "pretraining_code_checkpoints"
    / "FSM-Fine-Tuning-Dataset"
    / "artifacts"
    / "fsm_pretraining"
)
FSM_ADAPTER_PATH = Path(
    os.getenv(
        "FSM_PRETRAINED_PATH",
        FSM_ADAPTER_ROOT / os.getenv("FSM_PRETRAINED_CHECKPOINT", DEFAULT_FSM_CHECKPOINT),
    )
).expanduser()
FSM_BASE_MODEL_ID = os.getenv(
    "FSM_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

_fsm_model = None
_fsm_tokenizer = None
_fsm_device = None
_fsm_lock = Lock()

SYSTEM_PROMPT = (
    "You are an expert blockchain developer who writes production-grade Solidity "
    "smart contracts with detailed documentation and validation checks."
)


def _build_user_prompt(description: str, blockchain: str) -> str:
    return f"""
Generate a complete Solidity smart contract for the following description:
"{description}"

Requirements:
- Contract must target {blockchain}.
- Include events, modifiers, and comments explaining the logic.
- Validate inputs and follow best practices (access control, SafeMath when needed).
- After the code, list key clauses as `Clauses:` with numbered bullet points
  formatted as `Title: Description`.

Respond using this structure:
```solidity
// Solidity contract
```

Clauses:
1. Title: Description
2. Title: Description
"""


def _default_clauses() -> List["ContractClause"]:
    return [
        ContractClause(
            title="Core Requirements",
            description="Generated contract implements the functionality described by the user.",
        )
    ]


def _parse_contract_response(raw_text: str) -> tuple[str, List["ContractClause"]]:
    text = raw_text or ""
    lower_text = text.lower()

    contract_code = text.strip()
    start_token = "```solidity"
    start_idx = lower_text.find(start_token)
    if start_idx != -1:
        start_idx += len(start_token)
        end_idx = lower_text.find("```", start_idx)
        if end_idx != -1:
            contract_code = text[start_idx:end_idx].strip()
    else:
        generic_start = text.find("```")
        if generic_start != -1:
            generic_start += len("```")
            generic_end = text.find("```", generic_start)
            if generic_end != -1:
                contract_code = text[generic_start:generic_end].strip()

    clause_text = ""
    for marker in ("clauses:", "key clauses:", "contract clauses:", "important clauses:"):
        idx = lower_text.find(marker)
        if idx != -1:
            clause_text = text[idx + len(marker) :]
            break

    clauses: List[ContractClause] = []
    if clause_text:
        for line in clause_text.strip().splitlines():
            clean = line.strip()
            if not clean:
                continue

            clean = clean.lstrip("-*• ")
            while clean and (clean[0].isdigit() or clean[0] in {".", ")", "("}):
                clean = clean[1:].lstrip()

            if not clean:
                continue

            if ":" in clean:
                title, desc = clean.split(":", 1)
            elif " - " in clean:
                title, desc = clean.split(" - ", 1)
            else:
                title = f"Clause {len(clauses) + 1}"
                desc = clean

            clauses.append(
                ContractClause(title=title.strip(), description=desc.strip())
            )

    if not clauses:
        clauses = _default_clauses()

    if not contract_code:
        contract_code = text.strip()

    return contract_code, clauses


def _load_fsm_generation_stack():
    global _fsm_model, _fsm_tokenizer, _fsm_device
    with _fsm_lock:
        if _fsm_model is not None and _fsm_tokenizer is not None and _fsm_device is not None:
            return _fsm_model, _fsm_tokenizer, _fsm_device

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "The FSM fine-tuned generator requires `torch`, `transformers`, and `peft`."
            ) from exc

        if not FSM_ADAPTER_PATH.exists():
            raise FileNotFoundError(
                f"FSM adapter checkpoint not found at {FSM_ADAPTER_PATH}. "
                "Set FSM_PRETRAINED_PATH to the directory containing adapter_model.safetensors."
            )

        tokenizer = AutoTokenizer.from_pretrained(FSM_ADAPTER_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            FSM_BASE_MODEL_ID,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(
            base_model,
            FSM_ADAPTER_PATH,
        )
        model.to(device)
        model.eval()

        _fsm_model = model
        _fsm_tokenizer = tokenizer
        _fsm_device = device
        return _fsm_model, _fsm_tokenizer, _fsm_device

class ContractClause(BaseModel):
    title: str
    description: str

class SmartContract(BaseModel):
    contract_code: str = Field(..., description="The Solidity smart contract code")
    clauses: List[ContractClause]
    is_compilable: bool = Field(default=False)
    is_deployable: bool = Field(default=False)
    compiler_errors: Optional[str] = None
    deploy_errors: Optional[str] = None

# check if pydantic input is needed

@mcp.tool()
def generate_smart_contract(description: str, blockchain: str = "Ethereum") -> SmartContract:
    'This tool generates smart contracts for solidity based codes' # generate description
    # describe input too
    prompt = f"{SYSTEM_PROMPT}\n{_build_user_prompt(description, blockchain)}"

    llm = genai.GenerativeModel("gemini-2.0-flash")
    response = llm.generate_content(prompt)

    contract_code, clauses = _parse_contract_response(response.text)
    return SmartContract(
        contract_code=contract_code,
        clauses=clauses,
    )

@mcp.tool()
def generate_smart_contract_pretrained(
    description: str, blockchain: str = "Ethereum", max_new_tokens: int = 800
) -> SmartContract:
    """
    Generates a smart contract using the locally fine-tuned TinyLlama FSM checkpoint.
    """
    model, tokenizer, device = _load_fsm_generation_stack()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(description, blockchain)},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        chat_prompt = f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(description, blockchain)}"

    import torch

    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, input_ids.shape[-1] :]
    completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    contract_code, clauses = _parse_contract_response(completion)

    return SmartContract(
        contract_code=contract_code,
        clauses=clauses,
    )

@mcp.tool()
def check_deployability(contract: SmartContract, openzeppelin_path: str = None) -> SmartContract:
    """
    Compiles and attempts to deploy a Solidity contract.
    All compilation or deployment errors are c aptured in the SmartContract object.
    """
    contract.is_compilable = False
    contract.is_deployable = False
    contract.compiler_errors = None
    contract.deploy_errors = None

    try:
        # Install & set compiler version
        solcx.install_solc("0.8.20")
        solcx.set_solc_version("0.8.20")

        # Prepare import remappings if OpenZeppelin path is provided
        import_remappings = []
        if openzeppelin_path:
            openzeppelin_path = os.path.abspath(openzeppelin_path)  
            import_remappings.append(f"@openzeppelin/={openzeppelin_path}")

        # Compile Solidity source
        compiled = solcx.compile_source(
            contract.contract_code,
            output_values=["abi", "bin"],
            allow_paths=openzeppelin_path or "",
            import_remappings=import_remappings
        )
        contract.is_compilable = True

        # Extract ABI and bytecode
        _, contract_interface = compiled.popitem()
        abi = contract_interface["abi"]
        bytecode = contract_interface["bin"]

        # Setup Web3 in-memory blockchain
        w3 = Web3(Web3.EthereumTesterProvider())
        acct = w3.eth.accounts[0]
        Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

        # Detect constructor arguments and provide dummy values
        constructor_inputs = [c for c in abi if c.get("type") == "constructor"]
        args = []
        if constructor_inputs:
            for inp in constructor_inputs[0]["inputs"]:
                if inp["type"].startswith("uint"):
                    args.append(1)
                elif inp["type"].startswith("address"):
                    args.append(acct)
                else:
                    args.append(None)

        # Deploy contract
        try:
            tx_hash = Contract.constructor(*args).transact({"from": acct})
            tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if tx_receipt.contractAddress:
                contract.is_deployable = True
        except Exception as e:
            contract.deploy_errors = str(e)
            contract.is_deployable = False

    except solcx.exceptions.SolcError as e:
        contract.compiler_errors = str(e)
        contract.is_compilable = False
    except Exception as e:
        # Capture any other unexpected errors
        contract.compiler_errors = str(e)
        contract.is_compilable = False

    return contract

@mcp.tool()
def web_search(query: str, max_results: int) -> list[str]:
    """return spippets of text extracted from duck duck go search for the given
        query :  using DDGS search operators
        max_results: number of snippets to be returned, usually 5 - 20
    DDGS search operators Guidelines in the table below:
    Query example	Result
    cats dogs	Results about cats or dogs
    "cats and dogs"	Results for exact term "cats and dogs". If no results are found, related results are shown.
    cats -dogs	Fewer dogs in results
    cats +dogs	More dogs in results
    cats filetype:pdf	PDFs about cats. Supported file types: pdf, doc(x), xls(x), ppt(x), html
    dogs site:example.com	Pages about dogs from example.com
    cats -site:example.com	Pages about cats, excluding example.com
    intitle:dogs	Page title includes the word "dogs"
    inurl:cats	Page url includes the word "cats"
    """
    search_results = DDGS().text(query, max_results=max_results)
    return [f'{x["title"]}\n{x["body"]}\n{x["href"]}' for x in search_results]

if __name__ == "__main__":
    mcp.run(transport="stdio")
