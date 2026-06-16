import operator
from typing import List, TypedDict, Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    # inputs from csv. these wont change
    issue: str
    subject: str
    company: str

    # internal state
    decision: Literal["reply", "escalate", "out_of_scope"]
    all_retrieved_documents: List[Document]
    relevant_documents: List[Document]
    messages: Annotated[List[BaseMessage], add_messages]

    # outputs
    status: Literal["Replied", "Escalated"]
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]
    product_area: str
    response: str
    justification: str