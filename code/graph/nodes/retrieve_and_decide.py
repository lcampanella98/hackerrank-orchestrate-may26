"""
retrieve_and_decide.py
======================
LangGraph node: retrieve_and_decide
- Spawns a create_agent sub-agent with two tools
- Agent retrieves docs until it can decide: reply | escalate | out_of_scope
- Saves structured decision + full message history (no system prompt) to state
"""

from __future__ import annotations

import json
import textwrap
import traceback
from typing import List, Literal

from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from dotenv import load_dotenv

from graph.utils import format_documents, get_company_descriptor

load_dotenv()

import os

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import CHAT_MODEL, CHAT_MODEL
from graph.state import GraphState
from ingestion import get_vectorstore, get_collection_path

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

# ---------------------------------------------------------------------------
# Agent state – extends AgentState with our custom fields
# ---------------------------------------------------------------------------
class RetrieveAgentState(AgentState):
    """Agent-internal state that carries accumulated retrieved documents."""
    accumulated_docs: List[Document]


# ---------------------------------------------------------------------------
# Structured decision output
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field


class RetrievalDecision(BaseModel):
    """Final decision emitted by the retrieve-and-decide agent."""
    decision: Literal["reply", "escalate", "out_of_scope"] = Field(
        description=(
            "'reply' if the issue can be answered from documentation or is purely conversational "
            "'escalate' if it is sensitive/risky/requires a human, "
            "'out_of_scope' if it is entirely unrelated to the support domain."
        )
    )
    reasoning: str = Field(
        description="One-sentence rationale for the decision."
    )
    sources: List[str] = Field(
        description="The list of sources as paths, or an empty list if there are no sources."
    )

# ---------------------------------------------------------------------------
# Tool 1 – search_documentation
# ---------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def search_documentation(
    query: str,
    runtime: ToolRuntime[None, RetrieveAgentState],
) -> tuple[str, List[Document]]:
    """Search the support documentation vector store for passages relevant to the
    given query. Returns formatted excerpts with source paths. Call this whenever
    you need to look up information before making a decision.

    Args:
        query: The search query derived from the support issue.
    """
    # Retrieve the company collection from agent state (injected via context below)
    collection: str = runtime.state.get("collection", "")
    print(f"Searching documentation in collection {collection}, query: {query}")
    if not collection or collection == 'none':
        return "No relevant documentation found for this query.", []

    vs = get_vectorstore(collection)
    docs: List[Document] = vs.similarity_search(query, k=4)

    for doc in docs:
        print(f"found document: {doc.metadata['source']}")

    if not docs:
        return "No relevant documentation found for this query.", []

    # Build human-readable content for the model
    content = format_documents(docs)

    # Accumulate docs into agent state via Command (artifact side-channel)
    # The node will harvest them from the ToolMessage.artifact field later.
    return content, docs

def load_document_from_path(rel_path, collection) -> Document | None:
    base_dir = get_collection_path(collection)
    full_path = os.path.join(base_dir, rel_path)

    try:
        loader = TextLoader(full_path, encoding="UTF-8")
        loaded_docs = loader.load()
    except Exception as exc:
        error_details = traceback.format_exc()
        print(f"ERROR Loading Document: {rel_path}: {error_details}")
        return None

    # Override metadata so source is the relative path (not the absolute path)
    docs: List[Document] = []
    for d in loaded_docs:
        docs.append(Document(page_content=d.page_content, metadata={"source": rel_path}))

    return docs[0]


# ---------------------------------------------------------------------------
# Tool 2 – get_full_doc_source
# ---------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def get_full_doc_source(
    rel_path: str,
    runtime: ToolRuntime[None, RetrieveAgentState],
) -> tuple[str, List[Document]]:
    """Retrieve the full text of a support documentation file by its relative
    path (as seen in the 'Source' field of search results). Use this when a
    search excerpt is insufficient and you need the complete document.

    Args:
        rel_path: Relative path to the document, e.g. 'billing/refund-policy.md'
    """

    collection: str = runtime.state.get("collection", "")
    print(f"getting full doc source. Collection: {collection}, rel_path: {rel_path}")
    base_dir = get_collection_path(collection)
    full_path = os.path.join(base_dir, rel_path)

    if not os.path.exists(full_path):
        return f"File not found: {rel_path}", []

    try:
        loader = TextLoader(full_path, encoding="UTF-8")
        loaded_docs = loader.load()
    except Exception as exc:
        return f"Error loading {rel_path}: {exc}", []

    # Override metadata so source is the relative path (not the absolute path)
    docs: List[Document] = []
    for d in loaded_docs:
        docs.append(Document(page_content=d.page_content, metadata={"source": rel_path}))

    content = f"Full document [{rel_path}]:\n\n{docs[0].page_content}"

    return content, docs



# ---------------------------------------------------------------------------
# Tool limit middleware – after N tool calls, strip tools so the model is
# forced to emit a final structured response instead of calling another tool.
# ---------------------------------------------------------------------------
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware import ModelRequest, ModelResponse
from typing import Callable, Any


def make_tool_limit_middleware(max_tool_calls: int = 6) -> Callable:
    """
    Returns a @wrap_model_call middleware that removes all tools from the
    model request once `max_tool_calls` have been made.

    With tools=[]:
    - The model receives no tool schemas, so it cannot call another tool.
    - create_agent's response_format (RetrievalDecision) still applies, so
      the model is prompted to emit the structured final answer.
    - The agent loop recognises there are no tool calls in the response and
      terminates normally, populating `structured_response` correctly.
    """

    @wrap_model_call
    def tool_limit(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # Count how many tool calls have been made so far in this run
        tool_call_count = sum(
            len(getattr(msg, "tool_calls", []))
            for msg in request.state.get("messages", [])
            if hasattr(msg, "tool_calls")
        )

        if tool_call_count >= max_tool_calls:
            # Strip all tools → model must respond with final structured output
            request = request.override(tools=[])

        return handler(request)

    return tool_limit



# ---------------------------------------------------------------------------
# System prompt factory
# ---------------------------------------------------------------------------
def _build_system_prompt(company: str, issue: str, subject: str) -> str:
    return textwrap.dedent(f"""
        {get_company_descriptor(company)}

        A customer has submitted the following ticket:
        ---
        Subject : {subject or "(none)"}
        Issue   : {issue}
        ---

        Your job is to:
        1. Use `search_documentation` at least once to retrieve relevant support articles. Always try multiple different and varied search queries
        to make sure we find any relevant support documentation for the issue. THIS DOCUMENTATION MAY REFERENCE OTHER NEEDED DOCUMENTATION. In that case, use the tool again. 
        2. Use `get_full_doc_source` if to get the complete text of a document. Use this when a document is relevant to a question, just to make sure you have all the relevant context. 
        THIS DOCUMENTATION MAY REFERENCE OTHER NEEDED DOCUMENTATION. In that case, use the search_documentation tool again. 
        3. Once you have enough information to make a decision, output ONLY 1) the final decision, 2) a one-sentence reasoning, and 3) the sources that resolve the issue as their paths, or else indicate that there are no sources that resolve the issue. Output nothing else.  

        Decision rules:
        - **reply**       – The documentation contains enough information to answer safely. Or the issue is purely conversational such as "Thanks for the help"
        - **escalate**    – The issue is urgent, sensitive, or high-risk. An issue can be escalated even if the documentation base doesn't address it
        - **out_of_scope** – The issue is entirely unrelated to {company} products/services


        Be thorough but efficient. Do not hallucinate policies not found in the docs.
    """).strip()


# ---------------------------------------------------------------------------
# Helper – harvest all Document artifacts from ToolMessages in history
# ---------------------------------------------------------------------------
def _collect_docs_from_messages(messages: list) -> List[Document]:
    docs: List[Document] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.artifact:
            artifact = msg.artifact
            if isinstance(artifact, list):
                for item in artifact:
                    if isinstance(item, Document):
                        docs.append(item)
    return docs


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------
def node_retrieve_and_decide(state: GraphState) -> GraphState:
    """
    Spawns a create_agent sub-agent that:
      - has access to search_documentation and get_full_doc_source tools
      - is limited to max 6 tool calls via ToolLimitMiddleware
      - produces a structured RetrievalDecision as its final response

    Returns updates to GraphState:
      - messages        : full conversation (HumanMessage + AIMessages + ToolMessages)
      - decision        : 'reply' | 'escalate' | 'out_of_scope'
      - all_retrieved_documents : every Document surfaced by tool calls
    """
    company = state["company"]
    issue = state["issue"]
    subject = state.get("subject", "")
    collection = company.lower().strip()

    system_prompt = _build_system_prompt(company, issue, subject)

    # We inject the collection name into agent state so tools can read it
    # via runtime.state["collection"].  We piggy-back on a custom state key.
    class _AgentStateWithCollection(RetrieveAgentState):
        collection: str

    agent = create_agent(
        model=llm,
        tools=[search_documentation, get_full_doc_source],
        system_prompt=system_prompt,
        state_schema=_AgentStateWithCollection,
        middleware=[make_tool_limit_middleware(max_tool_calls=6)],
    )

    # Invoke the agent; seed state with collection name
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Issue: {issue}"}],
        "collection": collection,        # picked up by tools via runtime.state
        "accumulated_docs": [],
    })

    # -----------------------------------------------------------------------
    # Extract conversation messages (exclude system prompt – it's not in
    # messages, create_agent keeps it separate)
    # -----------------------------------------------------------------------
    agent_messages: list[BaseMessage] = result.get("messages", [])

    # -----------------------------------------------------------------------
    # Extract structured decision from the final agent response
    # -----------------------------------------------------------------------
    llm_with_structured = llm.with_structured_output(RetrievalDecision)
    structured_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are given an assistant's response in natural language and are expected to 
parse it into the required schema including the decision, the reasoning for the decision, and a list of sources as paths or an empty list if there are no sources. 
The decision should be one of "reply", "escalate" or "out_of_scope". 
        """),
        ("user", "Response: {response}")
    ])
    structured_chain = structured_prompt_template | llm_with_structured

    print(f"Retrieval agent raw response: {agent_messages[-1].content}")

    structured: RetrievalDecision = structured_chain.invoke({"response": agent_messages[-1].content})

    if structured and isinstance(structured, RetrievalDecision):
        decision = structured.decision
        justification = structured.reasoning
        relevant_docs = []
        if collection != 'none':
            for source in structured.sources:
                doc = load_document_from_path(source.strip(), collection)
                if doc is not None:
                    relevant_docs.append(doc)
    else:
        # Fallback
        decision = 'escalate'
        justification = 'There was an error in investigating the issue. Escalate to human. '
        relevant_docs = []

    # -----------------------------------------------------------------------
    # Collect all retrieved documents from ToolMessage artifacts
    # -----------------------------------------------------------------------
    all_docs = _collect_docs_from_messages(agent_messages)

    return {
        "messages": agent_messages,
        "decision": decision,
        "justification": justification,
        "all_retrieved_documents": all_docs,
        "relevant_documents": relevant_docs,
    }


# ---------------------------------------------------------------------------
# Fallback decision parser (last-resort if structured_response is missing)
# ---------------------------------------------------------------------------
def _parse_decision_fallback(
    messages: list,
) -> Literal["reply", "escalate", "out_of_scope"]:
    """Scan the last AI message content for a decision keyword."""
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = ""
            if isinstance(msg.content, str):
                text = msg.content.lower()
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "").lower()

            # Try JSON first
            try:
                data = json.loads(text)
                d = data.get("decision", "")
                if d in ("reply", "escalate", "out_of_scope"):
                    return d  # type: ignore[return-value]
            except (json.JSONDecodeError, TypeError):
                pass

            # Keyword scan
            if "out_of_scope" in text or "out of scope" in text:
                return "out_of_scope"
            if "escalate" in text:
                return "escalate"
            if "reply" in text:
                return "reply"

    # Default to escalate when uncertain
    return "escalate"

if __name__ == "__main__":
    state: GraphState = {
        "company": "HackerRank",
        "subject": "Test Active in the system",
        "issue": "I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system.",
    }
    result_state = node_retrieve_and_decide(state)
    from pprint import pprint

    pprint(result_state)