from langchain_core.documents import Document


def format_documents(documents: list[Document]) -> str:
    lines: list[str] = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content
        lines.append(f"[{i}] Source: {source}\n{snippet}")
    if len(lines) == 0:
        lines = ["No documents."]
    content = "\n\n".join(lines)
    return content

def get_company_descriptor(company: str) -> str:
    if company.lower() == 'none':
        return 'You are a general senior support triage specialist. '
    return f"You are a senior support triage specialist for {company}"