import textwrap
from typing import Literal

from dotenv import load_dotenv
from langchain_community.chat_models.cohere import get_role
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from pydantic import BaseModel, Field

from config import CHAT_MODEL, CHAT_MODEL
from graph.nodes.constants import RETRIEVE_AND_DECIDE, CLASSIFY_PRODUCT_AREA, \
    CLASSIFY_REQUEST_TYPE
from graph.nodes.retrieve_and_decide import node_retrieve_and_decide
from graph.state import GraphState
from graph.utils import get_company_descriptor, format_documents

load_dotenv()

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)


class RequestTypeAnswer(BaseModel):
    """The answer to the request type question"""
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"] = Field(
        description="The request type based on the issue and the relevant documents, "
                    'must be one of: "product_issue", "feature_request", "bug", "invalid"'
    )


def _get_system_prompt(company, subject, issue):
    # TODO add few-shot prompting
    return textwrap.dedent(f"""
        {get_company_descriptor(company)}

        A customer has submitted the following ticket:
        ---
        Subject : {subject or "(none)"}
        Issue   : {issue}
        ---

        Your job is to classify the issue into the most relevant request type based on the issue itself and relevant support documents. 
        The request type must be one of: "product_issue", "feature_request", "bug", "invalid"
        "product_issue" should be used when the issue is related to the support documents, but is neither a feature_request nor a bug.
        "feature_request" should be used when the user is requesting a new functionality
        "bug" should be used when the issue is indicating that there's currently a problem in the system that's unexpected 
        "invalid" should be the result otherwise 
        Reference the support documents if needed, but you may be able to classify based only on the issue. 
    """).strip()


def _get_prompt_template(company, subject, issue):
    return ChatPromptTemplate.from_messages([
        ("system", _get_system_prompt(company, subject, issue)),
        ("user", "Issue: {issue}\n\nDocuments: {documents}")
    ])


def node_classify_request_type(state: GraphState) -> GraphState:
    company = state["company"]
    subject = state["subject"]
    issue = state["issue"]
    relevant_documents = state["relevant_documents"]

    docs_content = format_documents(relevant_documents)

    llm_structured = llm.with_structured_output(RequestTypeAnswer)
    chain = _get_prompt_template(company, subject, issue) | llm_structured

    result: RequestTypeAnswer = chain.invoke({"documents": docs_content, "issue": issue})

    return {"request_type": result.request_type}


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
                 .add_node(CLASSIFY_REQUEST_TYPE, node_classify_request_type)
                 .add_edge(RETRIEVE_AND_DECIDE)
                 .add_edge(CLASSIFY_REQUEST_TYPE)
                 .add_edge(CLASSIFY_REQUEST_TYPE, END)
                 )
    graph = test_flow.compile()

    result_state = graph.invoke(state)

    from pprint import pprint

    pprint(result_state)
