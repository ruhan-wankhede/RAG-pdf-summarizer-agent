from vectorstore import ChromaVectorStore
from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv()

llm_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
store = ChromaVectorStore("WorldWar2.pdf", llm_client)

def get_prompts(query: str, context: str) -> tuple[str, str]:
    system_prompt = f"""You are an expert document summarizer.
    
                You will be given a user query and a set of retrieved document chunks that are relevant to that query. 
                Use only the information in the provided chunks to generate a focused, coherent, and concise summary that 
                directly answers or addresses the query.
                
                Avoid adding information not present in the context. Do not speculate.
                If there is no data relating to the query mention that.
    """
    user_prompt = f"""
                ## User Query:
                {query}
                
                ## Retrieved Chunks:
                {context}
                
                ## Your Task:
                Write a focused and accurate summary based only on the retrieved chunks that addresses the user’s query.
    """

    return system_prompt, user_prompt


def main():
    while True:
        print("\n---------------------------------------------------\n(Type q to quit)")
        topic = input("Enter a topic or a question relating to the document: ")

        if topic == "q":
            print("Goodbye.")
            break

        results = store.query(topic)
        context = "\n\n".join(results["documents"][0])

        system_prompt, user_prompt = get_prompts(topic, context)
        response = llm_client.chat.complete(model="mistral-small-latest",
                                   messages=[
                                       {"role": "system", "content": system_prompt},
                                       {"role": "user", "content": user_prompt}
                                   ])

        print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
