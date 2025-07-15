from pydantic import BaseModel
from typing import List, Annotated
from langgraph.graph.message import add_messages

# Langgraph State
class RAGState(BaseModel):
    user_query: str = ""
    retrieval_query: str = ""
    context: str = ""
    response: str = ""
    messages: Annotated[list, add_messages] = []