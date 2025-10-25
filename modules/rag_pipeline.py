from typing import Iterable, Optional
import logging

from modules.llm_wrapper import load_llm
from modules.prompt_engineering import create_prompt
from modules.memory_manager import get_memory
from modules.response_validator import validate_response
from langchain.chains import LLMChain
from langchain.schema import Document

logger = logging.getLogger(__name__)

class NSSRAGPipeline:
    """Simple RAG pipeline wrapper."""

    def __init__(self):
        self.llm = load_llm()
        self.prompt = create_prompt()
        self.memory = get_memory()

    def answer_query(self, query: str, context_docs: Optional[Iterable[Document]] = None) -> str:
        """
        Answer a user query using the LLM chain.

        Args:
            query: The user question.
            context_docs: Optional iterable of langchain Document objects to build context.

        Returns:
            Validated response string prefixed with a pipeline tag.
        """
        context = ""
        if context_docs:
            try:
                context = "\n".join(d.page_content for d in context_docs)
            except Exception as e:
                logger.debug("Failed to build context from documents: %s", e)
                context = ""

        chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)
        try:
            raw_response = chain.run({"question": query, "context": context})
        except Exception:
            logger.exception("LLMChain run failed")
            raise

        validated = validate_response(query, context, raw_response)
        tag = "NSS"
        return f"[{tag}] {validated}"
