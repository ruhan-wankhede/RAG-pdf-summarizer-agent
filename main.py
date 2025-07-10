import chromadb
import pymupdf
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

class ChromaVectorStore:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.api_key = os.getenv("MISTRAL_API_KEY")

        self.embed_client = Mistral(api_key=self.api_key)
        self.model = "mistral-embed"

        self.client = chromadb.PersistentClient(path="./chroma_mistral_api")
        self.collection = self.client.get_or_create_collection(name="pdf_chunks")

        self._process_and_store_documents()

    def _get_embeddings(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.embed_client.embeddings.create(model=self.model, inputs=batch)
            all_embeddings.extend([e.embedding for e in response.data])
        return all_embeddings

    def _process_and_store_documents(self) -> None:
        doc = pymupdf.open(self.pdf_path)
        pdf_text = "".join([page.get_text() for page in doc])

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        chunks = text_splitter.create_documents([pdf_text], metadatas=[{"source": "WorldWar2.pdf"}])
        chunks = [chunk.page_content for chunk in chunks]
        embeddings = self._get_embeddings(chunks)

        ids = [f"chunk-{i}" for i in range(len(chunks))]
        metadata = [{"source": self.pdf_path}] * len(chunks)

        self.collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadata)

    def query(self, text: str, k: int = 3):
        query_embedding = self._get_embeddings([text])[0]
        return self.collection.query(query_embeddings=[query_embedding], n_results=k)


if __name__ == "__main__":
    store = ChromaVectorStore("WorldWar2.pdf")

    # Optional: test query
    results = store.query("Who was hitler?")
    print("\n🔍 Query Results:")
    for i, doc in enumerate(results["documents"][0]):
        print(f"\n--- Chunk {i + 1} ---\n{doc[:300]}")  # show first 300 chars

