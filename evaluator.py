from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import List, Any

class RAGEvaluator:
    def __init__(self):
        self.eval_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    def evaluate_faithfulness(self, query: str, answer: str, source_docs: List[Any]) -> str:
        """
        Faithfulness check: For each answer, compare claims in the response to the retrieved chunks.
        Flag answers that contain information not present in the source.
        Returns: "Yes" if supported, "No" otherwise (or N/A if missing docs).
        """
        if not source_docs:
            return "N/A"
            
        context = "\n\n".join([doc.page_content for doc in source_docs])
        
        prompt = f"""You are an evaluator checking an AI assistant's answer for faithfulness to the provided context.
Compare the claims in the answer to the context. If the answer contains information that is not present in the context or hallucinates, answer "No". If all claims in the answer are supported by the context, answer "Yes".

Context: 
{context}

Question: {query}
Answer: {answer}

Is this answer supported by the provided context? (Respond with strictly Yes or No):"""

        try:
            response = self.eval_llm.invoke(prompt).content.strip()
            # Clean up response to just be Yes or No
            if "yes" in response.lower():
                return "Yes"
            elif "no" in response.lower():
                return "No"
            return response
        except Exception as e:
            return f"Error: {str(e)}"
