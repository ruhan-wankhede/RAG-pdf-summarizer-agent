from vectorstore import ChromaVectorStore
from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv()

topic = input("Enter a topic or a question relating to the document: ")

llm_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
store = ChromaVectorStore("WorldWar2.pdf", llm_client)

results = store.query(topic)
context = "\n\n".join(results["documents"][0])


system_prompt = f"""You are an expert document summarizer.

            You will be given a user query and a set of retrieved document chunks that are relevant to that query. 
            Use only the information in the provided chunks to generate a focused, coherent, and concise summary that 
            directly answers or addresses the query.
            
            Avoid adding information not present in the context. Do not speculate.
            If there is no data relating to the query mention that.
"""
user_prompt = f"""
            ## User Query:
            {topic}
            
            ## Retrieved Chunks:
            {context}
            
            ## Your Task:
            Write a focused and accurate summary based only on the retrieved chunks that addresses the user’s query.
"""

response = llm_client.chat.complete(model="mistral-medium-2505",
                           messages=[
                               {"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_prompt}
                           ])

print(response.choices[0].message.content)
