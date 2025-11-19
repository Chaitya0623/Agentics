from mcp.server.fastmcp import FastMCP

import os
from pydantic import BaseModel, Field
from typing import List, Optional
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

@mcp.tool(name="generate_smart_contract")
def generate_smart_contract(description: str, blockchain: str = "Ethereum") -> SmartContract:
    """
    Generate a Solidity smart contract from a natural-language description.

    Parameters:
        description: Text describing the contract’s purpose, rules, and behavior.
        blockchain: Target EVM-compatible chain (e.g., Ethereum, Polygon). Default: Ethereum.

    Output:
        A SmartContract object containing:
        - contract_code: The generated Solidity source code.
        - clauses: Key clauses extracted from the description.
        - is_compilable / is_deployable (initially False)
        - compiler_errors / deploy_errors (initially None)

    Notes:
        The contract is generated using an LLM and may require compilation or testing.
    """

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

@mcp.tool(name="validate_smart_contract")
def validate_smart_contract(contract: SmartContract, openzeppelin_path: str = None) -> SmartContract:
    """
    Compile and attempt to deploy a Solidity smart contract.

    Parameters:
        contract: A SmartContract object containing Solidity source code.
        openzeppelin_path: Optional local filesystem path to OpenZeppelin contracts
                           for import resolution (e.g., @openzeppelin/...).

    Process:
        - Installs and sets Solidity compiler (solc 0.8.20).
        - Compiles the contract, capturing compiler errors.
        - Attempts deployment on an in-memory Ethereum test chain (EthereumTester).
        - Auto-fills dummy values for constructor arguments.

    Output:
        The same SmartContract object, updated with:
        - is_compilable: True if compilation succeeded.
        - compiler_errors: Set if compilation failed.
        - is_deployable: True if deployment succeeded.
        - deploy_errors: Set if contract failed to deploy.
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

@mcp.tool(name="web_search")
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