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
        # We will use a fast/cheap LLM for guardrails where needed
        self.guardrail_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    # ------------------
    # PART A1: INPUT GUARDRAILS
    # ------------------
    def check_input_guardrails(self, query: str) -> Tuple[bool, str, str]:
        """
        Returns (is_passed, error_code, modified_query_or_message)
        """
        # 1. Query length limit
        if len(query) > 500:
            return False, "QUERY_TOO_LONG", "Your query is too long. Please limit it to 500 characters."

        # 2. PII detection (basic)
        has_pii, clean_query = self._detect_and_strip_pii(query)
        # Even if PII is detected, we strip it and continue unless we want to block it entirely.
        # Requirements: "strip them before processing and warn the user"
        # We'll handle the warning outside, but here we update the query. Let's return a special tuple or handle it.
        # Actually, let's keep it simple: return the modified query, but maybe raise a flag for PII warning.
        
        # 3. Off-topic detection
        if not self._is_on_topic(clean_query):
            return False, "OFF_TOPIC", "I can only answer questions about Nova Scotia driving rules."

        if has_pii:
            return True, "PII_DETECTED", clean_query

        return True, "NONE", clean_query

    def _detect_and_strip_pii(self, text: str) -> Tuple[bool, str]:
        has_pii = False
        # Phone: e.g., 902-555-0199
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        # License plate: ABC 1234
        plate_pattern = r'\b[A-Z]{3}\s\d{4}\b'
        
        for pattern in [phone_pattern, email_pattern, plate_pattern]:
            if re.search(pattern, text):
                has_pii = True
                text = re.sub(pattern, "[REDACTED]", text)
                
        return has_pii, text
        
    def _is_on_topic(self, query: str) -> bool:
        # We can ask the LLM if it's on topic
        prompt = f"""Is the following user query related to driving, road rules, licenses, or navigation?
        Answer with a single word: YES or NO.
        Query: {query}"""
        
        response = self.guardrail_llm.invoke(prompt).content.strip().upper()
        return "YES" in response
