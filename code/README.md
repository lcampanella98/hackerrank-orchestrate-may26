## Project Setup
1. go into the code/ directory
1. Set .env file with valid OPENAI_API_KEY
1. make sure the uv package manager is installed (see https://docs.astral.sh/uv/getting-started/installation/)
1. run `uv sync`
1. load/source the created virtual environment in your terminal 
1. run `python ingestion.py`. this will create local chroma vectorstores with the documentation corpuses
1. run `python main.py`. this will run the agent against the support_tickets and write to the output file

## Approach Overview
At a high level, the approach uses the LangChain and LangGraph frameworks to 
orchestrate the agent. LangGraph models AI agentic systems as a state
graph with many useful features including parallel execution. 

The graph begins with the retrieve_and_decide node, which creates an
agent that uses tools looks up relevant documentation, decide how to 
respond if at all, and provide a justification and list of relevant sources.
Importantly, the relevant sources in the response only include those 
documents needed to resolve the issue, not all the documents 
looked-up in the tool calls. This is important for later steps. 

The graph then branches to three nodes in parallel: generate_response,
classify_product_area, and classify_request_type. 

The generate_response node takes the issue and relevant documents and generates
the response field. 

The classify_product_area node takes the issue and relevant sources in and 
returns the product area. 

The classify_request_type takes the issue and relevant documents and
returns the request type. 

Then the graph execution ends and the result contains all the 
output fields. 