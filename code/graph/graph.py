from langgraph.constants import END
from langgraph.graph import StateGraph

from graph.nodes.classify_product_area import node_classify_product_area
from graph.nodes.classify_request_type import node_classify_request_type
from graph.nodes.constants import RETRIEVE_AND_DECIDE, GENERATE_RESPONSE, \
    CLASSIFY_PRODUCT_AREA, CLASSIFY_REQUEST_TYPE
from graph.nodes.generate_response import node_generate_response
from graph.nodes.retrieve_and_decide import node_retrieve_and_decide
from graph.state import GraphState

workflow = StateGraph(GraphState)
# nodes
workflow.add_node(RETRIEVE_AND_DECIDE, node_retrieve_and_decide)
workflow.add_node(GENERATE_RESPONSE, node_generate_response)
workflow.add_node(CLASSIFY_PRODUCT_AREA, node_classify_product_area)
workflow.add_node(CLASSIFY_REQUEST_TYPE, node_classify_request_type)

# edges
workflow.set_entry_point(RETRIEVE_AND_DECIDE)
workflow.add_edge(RETRIEVE_AND_DECIDE, GENERATE_RESPONSE)
workflow.add_edge(RETRIEVE_AND_DECIDE, CLASSIFY_PRODUCT_AREA)
workflow.add_edge(RETRIEVE_AND_DECIDE, CLASSIFY_REQUEST_TYPE)
workflow.add_edge(GENERATE_RESPONSE, END)
workflow.add_edge(CLASSIFY_PRODUCT_AREA, END)
workflow.add_edge(CLASSIFY_REQUEST_TYPE, END)

app = workflow.compile()

