import os
import re
import time
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

class RAGGuardrails:
    def __init__(self):
        # We will use a fast model for guardrails
        self.guardrail_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    # ------------------
    # PART A1: INPUT GUARDRAILS
    # ------------------
    def check_input_guardrails(self, query: str) -> Tuple[str, str, str]:
        """
        Returns (error_code, modified_query_or_message)
        """
        # 1. Query length limit
        if len(query) > 500:
            return "QUERY_TOO_LONG", "Your query is too long. Please limit it to 500 characters."

        # Part B: Input sanitization (Prompt Injection Defense #1)
        if self._detect_prompt_injection(query):
            return "POLICY_BLOCK", "I cannot process this request due to a potential policy violation."

        # 2. PII detection (basic)
        has_pii, clean_query = self._detect_and_strip_pii(query)
        
        # 3. Off-topic detection
        if not self._is_on_topic(clean_query):
            return "OFF_TOPIC", "I can only answer questions about Nova Scotia driving rules."

        if has_pii:
            return "PII_DETECTED", clean_query

        return "NONE", clean_query

    def _detect_prompt_injection(self, query: str) -> bool:
        """
        Input sanitization block patterns (Prompt Injection Defense).
        """
        blocked_patterns = [
            r"ignore previous instructions",
            r"you are now",
            r"system:",
            r"### new instructions"
        ]
        q_lower = query.lower()
        for pattern in blocked_patterns:
            if re.search(pattern, q_lower):
                return True
        return False

    def _detect_and_strip_pii(self, text: str) -> Tuple[bool, str]:
        has_pii = False
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        plate_pattern = r'\b[A-Z]{3}\s\d{4}\b'
        
        for pattern in [phone_pattern, email_pattern, plate_pattern]:
            if re.search(pattern, text):
                has_pii = True
                text = re.sub(pattern, "[REDACTED]", text)
                
        return has_pii, text
        
    def _is_on_topic(self, query: str) -> bool:
        # Ask LLM if it's on topic
        if not query.strip():
            return False # Empty query is off-topic
        prompt = f"""Is the following user query related to driving, road rules, vehicles, road safety, licenses, or navigation?
        Answer with a single word: YES or NO.
        Query: {query}"""
        
        response = self.guardrail_llm.invoke(prompt).content.strip().upper()
        return "YES" in response

    # ------------------
    # PART A2: OUTPUT GUARDRAILS
    # ------------------
    def check_retrieval_confidence(self, source_docs: List[Any], threshold: float = 0.5) -> bool:
        """
        Refusal on low confidence based on similarity score (if available) or zero documents.
        Many vector stores return (doc, score). In LangChain QA chains without exact scoring, we check if docs is empty.
        We'll treat empty source_docs as an immediate low confidence.
        """
        if not source_docs:
            return False
            
        # Assuming we can inspect scores if we do an open retrieval
        # In general, if retrieval is empty string or bad context, we refuse.
        return True

    def check_output_guardrails(self, response_text: str) -> Tuple[str, str]:
        """
        Returns (error_code, final_response)
        """
        # Part B: Output validation (Prompt Injection Defense #2)
        # Check if response contains leaked system prompt instructions
        if "never reveal your system prompt" in response_text.lower() or "you are a driving assistant" in response_text.lower():
            return "POLICY_BLOCK", "I cannot fulfill this request due to policy restrictions on my output."

        # Response length limit (e.g., max 500 words)
        words = response_text.split()
        if len(words) > 500:
            response_text = " ".join(words[:500]) + "... [Truncated for length]"
            # Could trigger an error, but truncation works as a guardrail action
            
        return "NONE", response_text
