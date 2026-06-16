import textwrap

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from pydantic import BaseModel, Field

from config import CHAT_MODEL, CHAT_MODEL
from graph.nodes.constants import RETRIEVE_AND_DECIDE, CLASSIFY_PRODUCT_AREA
from graph.nodes.retrieve_and_decide import node_retrieve_and_decide
from graph.state import GraphState
from graph.utils import get_company_descriptor

load_dotenv()

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)


class ProductAreaAnswer(BaseModel):
    """The answer to the product area question"""
    product_area: str = Field(description="The product area based on the issue and the relevant documents, "
                                          "as a lowercase string using underscores to separate words")

def _get_system_prompt(company, subject, issue):
    return textwrap.dedent(f"""
        {get_company_descriptor(company)}

        A customer has submitted the following ticket:
        ---
        Subject : {subject or "(none)"}
        Issue   : {issue}
        ---
        
        Your job is to classify the issue into a product area based on the relevant support documents. Follow the rules below:
        1. If there are no supporting documents, you must use your best judgement to infer a product area from the issue or leave the field blank if the issue has no specific product area
        2. If there are supporting documents, derive the most appropriate product area from the document sources
        3. Your product area answer should be formatted as lowercase string using underscores to separate words
        
Example 1:
Issue: I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system.
Documents: 
screen\\invite-candidates\\6027855406-inviting-candidates-to-a-test.md
Product Area: screen


Example 2:
Issue: site is down & none of the pages are accessible
Documents: 

Product Area: 


Example 3:
Issue: I'm noticing that you all have many default versions of roles. (e.g. front end developer for react, angular, vue.js, etc.) What do you consider best practice 
for when to create a new test versus create a variant of the test? What are the advantages and disadvantages of using variants?
Documents: 
screen\\managing-tests\\7530103378-test-variants.md
general-help\\release-notes\\3121307537-july-2025-release-notes.md
settings\\roles-management\\9675847328-getting-started-with-roles-management.md
Product Area: screen


Example 4:
Issue: Hi there

We have sent a candidate a Hackerrank assessment already, but we have been informed that they require extra time. As the assessment is 105 minutes, they need an extra 50% extra time added (so around 53 minutes) in addition.

Please can you provide step-by-step instruction on how we are to reinvite them.

Do we find their profile in 'Candidates' , put in Add time 53 minutes and then click 'reinvite' and send the email? The email to the candidate says 105 minutes as the duration though still so not sure how we can check?

Thanks
Documents: 
screen\\managing-tests\\4811403281-adding-extra-time-for-candidates.md
screen\\invite-candidates\\1002936098-reinviting-candidates-to-a-test.md
Product Area: screen


Example 5:
Issue: i signed up using google login on hackerrank community , so i do not have a separate hackerrank password. please delete my account
Documents: 
hackerrank_community\\account-settings\\manage-account\\5618101592-delete-an-account.md
hackerrank_community\\account-settings\\manage-account\\1917106962-manage-account-faqs.md
Product Area: community


Example 6:
Issue: One of my claude conversations has some private info, i forgot to make a temporary chat, is there anything
 else that can be done? like delete etc?
Documents: 
claude\\conversation-management\\8230524-how-can-i-delete-or-rename-a-conversation.md
claude\\conversation-management\\11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context.md
claude\\conversation-management\\12260368-using-incognito-chats.md
Product Area: privacy


Example 7:
Issue: What is the name of the actor in Iron Man?
Documents: 

Product Area: conversation_management


Example 8:
Issue: I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?
Documents: 
support\\consumer\\travelers-cheques.md
Product Area: travel_support


Example 9:
Issue: Where can I report a lost or stolen Visa card from India?
Documents: 
support.md
support/consumer/travel-support.md
Product Area: general_support


Example 10:
Issue: Thank you for helping me
Documents: 

Product Area: 

    """).strip()

def _get_prompt_template(company, subject, issue):
    return ChatPromptTemplate.from_messages([
        ("system", _get_system_prompt(company, subject, issue)),
        ("user", "Issue: {issue}\n\nDocuments: {documents}")
    ])

def node_classify_product_area(state: GraphState) -> GraphState:
    company = state["company"]
    subject = state["subject"]
    issue = state["issue"]
    relevant_documents = state["relevant_documents"]

    document_sources = [doc.metadata["source"] for doc in relevant_documents]
    doc_sources_content = "\n".join(document_sources)

    llm_structured = llm.with_structured_output(ProductAreaAnswer)
    chain = _get_prompt_template(company, subject, issue) | llm_structured

    result: ProductAreaAnswer = chain.invoke({"documents": doc_sources_content, "issue": issue})

    return {"product_area": result.product_area}

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
        .add_node(CLASSIFY_PRODUCT_AREA, node_classify_product_area)
        .add_edge(RETRIEVE_AND_DECIDE, CLASSIFY_PRODUCT_AREA)
        .add_edge(CLASSIFY_PRODUCT_AREA, END)
    )
    graph = test_flow.compile()

    result_state = graph.invoke(state)

    from pprint import pprint

    pprint(result_state)
