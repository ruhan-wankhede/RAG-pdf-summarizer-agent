import chromadb
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral

class ChromaVectorStore:
    def __init__(self, pdf_path: str, client: Mistral):
        self.pdf_path = pdf_path
        self.embed_client = client
        self.model = "mistral-embed"

        self.client = chromadb.EphemeralClient()

        if "pdf_chunks" in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection("pdf_chunks")

        self.collection = self.client.create_collection(name="pdf_chunks")

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

        chunks = text_splitter.create_documents([pdf_text], metadatas=[{"source": self.pdf_path}])
        chunks = [chunk.page_content for chunk in chunks]
        embeddings = self._get_embeddings(chunks)

        ids = [f"chunk-{i}" for i in range(len(chunks))]
        metadata = [{"source": self.pdf_path}] * len(chunks)

        self.collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadata)

    def query(self, text: str, k: int = 5):
        query_embedding = self._get_embeddings([text])[0]
        return self.collection.query(query_embeddings=[query_embedding], n_results=k)

