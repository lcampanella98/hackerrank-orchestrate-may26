import textwrap

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END

from config import CHAT_MODEL, CHAT_MODEL
from graph.nodes.constants import RETRIEVE_AND_DECIDE, GENERATE_RESPONSE
from graph.nodes.retrieve_and_decide import node_retrieve_and_decide
from graph.state import GraphState
from graph.utils import format_documents, get_company_descriptor

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

def _build_system_prompt(company: str, issue: str, subject: str) -> str:
    return textwrap.dedent(f"""
        {get_company_descriptor(company)}

        A customer has submitted the following ticket:
        ---
        Subject : {subject or "(none)"}
        Issue   : {issue}
        ---

        Your job is to use ONLY the provided support documents to generate a helpful response for the issue. The response
        should be as concise as possible while still providing all needed details. 
        Be concise by default. Provide clear, direct answers using the fewest words necessary. Avoid unnecessary elaboration, repetition, or filler. 

        Do not hallucinate policies not found in the provided support documents.
    """).strip()

def _build_prompt_template(company: str, issue: str, subject: str):
    return ChatPromptTemplate.from_messages([
        ("system", _build_system_prompt(company, issue, subject)),
        ("user", "Issue: {issue}\n\nDocuments: {documents}\n\nResponse: ")
    ])

def node_generate_response(state: GraphState) -> GraphState:
    company = state["company"]
    subject = state["subject"]
    issue = state["issue"]
    relevant_documents = state["relevant_documents"]
    decision = state["decision"]

    # compute status
    if decision == 'reply' or decision == 'out_of_scope':
        status = 'Replied'
    else:
        status = 'Escalated'

    if decision == 'out_of_scope':
        return {
            "decision": decision,
            "response": "I am sorry, this is out of scope from my capabilities",
            "status": status,
        }
    if decision == 'escalate':
        return {
            "response": "Escalate to a human",
            "status": status,
        }
    # Reply
    # Build human-readable content for the model
    document_content = format_documents(relevant_documents)

    chain = _build_prompt_template(company, issue, subject) | llm | StrOutputParser()
    response_text = chain.invoke({"documents": document_content, "issue": issue})
    return {
        "response": response_text,
        "status": status,
    }

if __name__ == "__main__":
    state: GraphState = {
        "company": "HackerRank",
        "subject": "Test Active in the system",
        "issue": "I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system.",
    }
    from langgraph.graph import StateGraph
    test_flow = (StateGraph(GraphState)
        .set_entry_point(RETRIEVE_AND_DECIDE)
        .add_node(RETRIEVE_AND_DECIDE, node_retrieve_and_decide)
        .add_node(GENERATE_RESPONSE, node_generate_response)
        .add_edge(RETRIEVE_AND_DECIDE)
        .add_edge(GENERATE_RESPONSE)
        .add_edge(GENERATE_RESPONSE, END)
    )
    graph = test_flow.compile()

    result_state = graph.invoke(state)

    from pprint import pprint

    pprint(result_state)
    pprint(result_state["response"])