# DocMind-RAG - Assignment 3 (Security & Evaluation)

This project is an enhanced RAG system focused on security, guardrails, and evaluation. It uses LangChain, ChromaDB, HuggingFace embeddings, and Google Gemini.

## Assignment 3 Implementation Details

### 1. Prompt Injection Defenses (Implemented 4)
- **System Prompt Hardening**: The system prompt explicitly instructs the LLM to only answer driving questions, treat retrieved data as untrusted, and never reveal its instructions.
- **Input Sanitization**: Scans user queries for adversarial patterns (e.g., "ignore previous instructions", "you are now") and blocks them.
- **Instruction-Data Separation**: Context chunks are clearly delimited using `<retrieved_context>` tags to prevent the LLM from confusing data with instructions.
- **Output Validation**: Validates the LLM's response to ensure it doesn't contain leaked instructions or restricted patterns.

### 2. Guardrails (Implemented All)
- **Input**: Query length limit (500 chars), Off-topic detection, and PII detection (Redacts phone numbers, emails, and license plates).
- **Output**: Refusal on low confidence (Similarity threshold < 0.3) and Response length capping (500 words).
- **Execution**: 30-second timeout and structured error handling (Taxonomy: `QUERY_TOO_LONG`, `OFF_TOPIC`, `PII_DETECTED`, `RETRIEVAL_EMPTY`, `LLM_TIMEOUT`, `POLICY_BLOCK`).

### 3. Evaluation Metric
- **Faithfulness Check**: Every generated answer is evaluated by a separate LLM call to verify if the claims are supported by the retrieved context.

### 4. Interesting Findings
- **Gemini Latency**: During testing, several queries exceeded the 30-second timeout limit. This highlights the importance of execution limits in production to prevent runaway costs/wait times.
- **Guardrail Effectiveness**: The input sanitization layer successfully blocked all direct prompt injection attempts before they even reached the retrieval stage.
- **PII Redaction**: Basic regex-based PII detection is fast and effective for common patterns like phone numbers, preserving privacy while allowing the query to proceed.

## Setup

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Environment Variables**:
    - Add your `GOOGLE_API_KEY` to a `.env` file.

3.  **Add Documents**:
    - Ensure `data/DH-Chapter2.pdf` is present.

4.  **Run and Test**:
    ```bash
    uv run python rag_system.py
    ```
    - Check the **Logging Dashboard** printed in the terminal.
    - View detailed results in `output/results.txt`.

