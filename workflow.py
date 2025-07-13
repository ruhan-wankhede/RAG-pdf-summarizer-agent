from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from mistralai import Mistral
from dotenv import load_dotenv
import os
from vectordb import ChromaVectorStore
from prompts import Prompts
from state import RAGState

load_dotenv()


class Workflow:
    def __init__(self, pdf_path: str):
        self.llm = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.vectorstore = ChromaVectorStore(pdf_path, self.llm)
        self.prompts = Prompts()
        self.workflow = self._build_workflow()
        self.thread_config = {"configurable": {"thread_id": "1"}}

    def _to_mistral_format(self, messages: list) -> list[dict]:
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system"
        }
        return [{"role": role_map.get(m.type, "user"), "content": m.content} for m in messages]

    def _build_workflow(self):
        graph = StateGraph(RAGState)

        graph.add_node("generate_query", self._generate_retrieval_query_step)
        graph.add_node("retrieve_context", self._retrieve_context_step)
        graph.add_node("generate_summary", self._generate_summary_step)

        graph.set_entry_point("generate_query")
        graph.add_edge("generate_query", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_summary")
        graph.add_edge("generate_summary", END)

        return graph.compile()

    def _generate_retrieval_query_step(self, state: RAGState) -> dict:
        messages = [
            SystemMessage(content=self.prompts.SYSTEM_RETRIEVAL_QUERY),
            HumanMessage(content=state.user_query)
        ]

        response = self.llm.chat.complete(model="mistral-small-latest", messages=self._to_mistral_format(messages))

        retrieval_query = response.choices[0].message.content.strip()
        return {"retrieval_query": retrieval_query}

    def _query_pdf(self, query: str, k: int = 5) -> str:
        results = self.vectorstore.query(query, k)
        return "\n".join(results["documents"][0])

    def _retrieve_context_step(self, state: RAGState) -> dict:
        context = self._query_pdf(state.retrieval_query)
        return {"context": context}

    def _generate_summary_step(self, state: RAGState) -> dict:
        system_prompt, user_prompt = self.prompts.get_generation_prompts(query=state.retrieval_query, context=state.context)

        new_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.chat.complete(model="mistral-small-latest", messages=self._to_mistral_format(new_messages))
        reply = response.choices[0].message.content.strip()

        updated_messages = state.messages + [
            HumanMessage(content=state.user_query),  # Original user query
            AIMessage(content=reply)  # AI response
        ]
        return {"response": reply, "messages": updated_messages}

    def run_workflow(self, state: RAGState) -> RAGState:
        return RAGState(**self.workflow.invoke(state))


