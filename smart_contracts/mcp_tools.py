import os
import sys
from pathlib import Path
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
    prompt = f"""
    You are an expert blockchain developer.
    Generate a complete Solidity smart contract for the following description:
    "{description}"

    Requirements:
    - Contract must be valid for {blockchain}.
    - Include functions, modifiers, and comments.
    - Provide only the Solidity code and a list of key clauses (title + description).
    """

    llm = genai.GenerativeModel("gemini-2.0-flash")
    response = llm.generate_content(prompt)

    # Return structured result
    return SmartContract(
        contract_code=response.text.strip(),
        clauses=[
            ContractClause(title="Example Clause", description="Core logic or rule implemented here.")
        ],
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
