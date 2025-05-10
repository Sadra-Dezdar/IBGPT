"""Multi-agent system orchestrator."""

import os
from dataclasses import dataclass
from typing import Dict, Any, List
import chromadb
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from tools.context_retriever import retrieve_context
from tools.ia_feedback import provide_ia_feedback
from tools.exam_handler import solve_exam_question

@dataclass
class MultiAgentDeps:
    """Dependencies for multi-agent system."""
    chroma_client: chromadb.PersistentClient
    embedding_model: str = "all-MiniLM-L6-v2"

class MultiAgentSystem:
    """Orchestrates multiple agents for IB queries."""
    
    def __init__(self):
        # Get Ollama host from environment or use default
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Initialize agents with Ollama models via OpenAI compatibility
        self.fast_agent = Agent(
            OpenAIModel(
                "qwen3:latest",
                provider=OpenAIProvider(
                    base_url=f"{ollama_host}/v1",
                    api_key="ollama"
                )
            ),
            deps_type=MultiAgentDeps,
            system_prompt="""You are a query router for IB student questions. 
            Classify queries and extract key information for retrieval.
            Output JSON format with: query_type, subject, level, search_terms."""
        )
        
        # Use llama3.2 for RAG agent since it supports tools
        self.rag_agent = Agent(
            OpenAIModel(
                "llama3.2:latest",  # Changed from deepseek
                provider=OpenAIProvider(
                    base_url=f"{ollama_host}/v1",
                    api_key="ollama"
                )
            ),
            deps_type=MultiAgentDeps,
            system_prompt="""You are an expert IB educator. Use provided context 
            to give accurate, helpful responses following IB standards.""",
            tools=[retrieve_context, provide_ia_feedback, solve_exam_question]
        )
        
        # Use llama3.2 for consensus agent as well
        self.consensus_agent = Agent(
            OpenAIModel(
                "llama3.2:latest",  # Changed from deepseek
                provider=OpenAIProvider(
                    base_url=f"{ollama_host}/v1",
                    api_key="ollama"
                )
            ),
            deps_type=MultiAgentDeps,
            system_prompt="""Review and refine responses for accuracy and clarity."""
        )
    
    async def process_query(self, query: str, deps: MultiAgentDeps) -> str:
        """Process user query through the multi-agent pipeline."""
        
        # Step 1: Classify query with Qwen (no_think mode)
        classification_prompt = f"/no_think Classify this IB student query and return JSON: {query}"
        
        try:
            classification_result = await self.fast_agent.run(
                classification_prompt, 
                deps=deps
            )
            
            # Parse classification
            classification = self._parse_classification(query, classification_result.data)
        except Exception as e:
            print(f"Classification error: {e}")
            # Fallback classification
            classification = self._fallback_classification(query)
        
        # Step 2: Process with RAG agent
        try:
            rag_response = await self.rag_agent.run(
                query,
                deps=deps,
                context={"classification": classification}
            )
            response_text = rag_response.data
        except Exception as e:
            print(f"RAG error: {e}")
            response_text = f"I encountered an error processing your query: {e}"
        
        # Step 3: Review and refine
        try:
            final_response = await self.consensus_agent.run(
                f"Review and refine this response:\n{response_text}\n\nOriginal query: {query}",
                deps=deps
            )
            return final_response.data
        except Exception as e:
            print(f"Consensus error: {e}")
            # Return RAG response if consensus fails
            return response_text
    
    def _parse_classification(self, query: str, classification_text: str) -> Dict[str, Any]:
        """Parse classification result."""
        # This is a simplified parser - in production, you'd parse actual JSON
        classification = {
            "query_type": "general_info",
            "subject": None,
            "level": None,
            "search_terms": query.split()[:5]
        }
        
        # Try to extract info from classification text
        text_lower = classification_text.lower()
        
        if "ia" in text_lower or "internal assessment" in text_lower:
            classification["query_type"] = "ia_feedback"
        elif "exam" in text_lower or "paper" in text_lower:
            classification["query_type"] = "exam_question"
        
        # Extract subject
        subjects = ["mathematics aa", "mathematics ai", "physics", "chemistry"]
        for subject in subjects:
            if subject in text_lower:
                classification["subject"] = subject.title()
                break
        
        # Extract level
        if "hl" in text_lower:
            classification["level"] = "HL"
        elif "sl" in text_lower:
            classification["level"] = "SL"
        
        return classification
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Fallback classification based on keywords."""
        query_lower = query.lower()
        
        classification = {
            "query_type": "general_info",
            "subject": None,
            "level": None,
            "search_terms": query.split()[:5]
        }
        
        # Detect query type
        if any(term in query_lower for term in ["ia", "internal assessment"]):
            classification["query_type"] = "ia_feedback"
        elif any(term in query_lower for term in ["exam", "question", "paper", "solve"]):
            classification["query_type"] = "exam_question"
        
        # Extract subject
        if "math" in query_lower:
            if "aa" in query_lower or "analysis" in query_lower:
                classification["subject"] = "Mathematics AA"
            elif "ai" in query_lower or "applications" in query_lower:
                classification["subject"] = "Mathematics AI"
            else:
                classification["subject"] = "Mathematics AA"  # Default
        
        # Extract level
        if "hl" in query_lower or "higher level" in query_lower:
            classification["level"] = "HL"
        elif "sl" in query_lower or "standard level" in query_lower:
            classification["level"] = "SL"
        
        return classification