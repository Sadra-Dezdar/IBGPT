"""Multi-agent system orchestrator without tool dependencies."""

import os
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import chromadb
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from core.retrieval import multi_collection_search, retrieve_documents
from config.collections import DOC_TYPE_TO_COLLECTION

@dataclass
class MultiAgentDeps:
    """Dependencies for multi-agent system."""
    chroma_client: chromadb.PersistentClient
    embedding_model: str = "all-MiniLM-L6-v2"

class MultiAgentSystem:
    """Orchestrates multiple agents for IB queries without using tools."""
    
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
            Analyze the user's query and output a JSON response with:
            {
                "query_type": "general_info", // or "ia_feedback" or "exam_question"
                "subject": "Mathematics AA", // or other subject, null if not specific
                "level": "HL", // or "SL" or null
                "search_terms": ["term1", "term2", ...] // key search terms
            }"""
        )
        
        # Use deepseek for RAG agent without tools
        self.rag_agent = Agent(
            OpenAIModel(
                "deepseek-r1:14b-qwen-distill-q4_K_M",
                provider=OpenAIProvider(
                    base_url=f"{ollama_host}/v1",
                    api_key="ollama"
                )
            ),
            deps_type=MultiAgentDeps,
            system_prompt="""You are an expert IB educator. Using the provided context, 
            give accurate, helpful responses following IB standards. Base your answers 
            on the retrieved documents and cite them when appropriate."""
        )
        
        # Use deepseek for consensus agent
        self.consensus_agent = Agent(
            OpenAIModel(
                "deepseek-r1:14b-qwen-distill-q4_K_M",
                provider=OpenAIProvider(
                    base_url=f"{ollama_host}/v1",
                    api_key="ollama"
                )
            ),
            deps_type=MultiAgentDeps,
            system_prompt="""Review and refine responses for accuracy and clarity.
            Ensure the response follows IB terminology and standards."""
        )
    
    def _parse_thinking(self, text: str) -> Tuple[str, str]:
        """Parse out thinking content from response."""
        # Find all <think> content
        thinking_pattern = r'<think>(.*?)</think>'
        thinking_matches = re.findall(thinking_pattern, text, re.DOTALL)
        
        # Remove thinking from main content
        main_content = re.sub(thinking_pattern, '', text, flags=re.DOTALL).strip()
        
        # Join all thinking content
        thinking_content = '\n'.join(thinking_matches) if thinking_matches else ''
        
        return main_content, thinking_content
    
    async def process_query(self, query: str, deps: MultiAgentDeps) -> Dict[str, str]:
        """Process user query through the multi-agent pipeline."""
        
        # Step 1: Classify query with Qwen
        try:
            classification_result = await self.fast_agent.run(
                f"Analyze this IB student query and return JSON classification: {query}", 
                deps=deps
            )
            
            # Parse classification
            classification = self._parse_json_response(classification_result.data)
            if not classification:
                classification = self._fallback_classification(query)
                
        except Exception as e:
            print(f"Classification error: {e}")
            classification = self._fallback_classification(query)
        
        # Step 2: Retrieve context manually
        try:
            context = await self._retrieve_context(deps, query, classification)
        except Exception as e:
            print(f"Retrieval error: {e}")
            context = "No relevant documents found."
        
        # Step 3: Process with RAG agent using retrieved context
        try:
            prompt = f"""
Query: {query}

Retrieved Context:
{context}

Based on the above context, provide a comprehensive answer to the query.
"""
            rag_response = await self.rag_agent.run(prompt, deps=deps)
            response_text = rag_response.data
        except Exception as e:
            print(f"RAG error: {e}")
            response_text = f"I encountered an error processing your query: {e}"
        
        # Step 4: Review and refine
        try:
            final_prompt = f"""
Original query: {query}

Response to review:
{response_text}

Please review and refine this response to ensure accuracy and clarity.
"""
            final_response = await self.consensus_agent.run(final_prompt, deps=deps)
            final_text = final_response.data
        except Exception as e:
            print(f"Consensus error: {e}")
            final_text = response_text
        
        # Parse thinking from final response
        main_response, thinking = self._parse_thinking(final_text)
        
        return {
            "response": main_response,
            "thinking": thinking
        }
    
    async def _retrieve_context(self, deps: MultiAgentDeps, query: str, 
                               classification: Dict[str, Any]) -> str:
        """Manually retrieve context based on classification."""
        
        # Build metadata filter
        metadata_filter = {}
        if classification.get("subject"):
            metadata_filter["subject"] = classification["subject"]
        if classification.get("level"):
            metadata_filter["level"] = classification["level"]
        
        # Determine which collections to search
        query_type = classification.get("query_type", "general_info")
        
        if query_type == "ia_feedback":
            collections = ["ia_guides", "ia_examples"]
        elif query_type == "exam_question":
            collections = ["mark_schemes", "general_info"]  # No past_papers collection
        else:
            collections = ["ib_general", "syllabus"]
        
        # Search across collections
        all_results = []
        search_query = " ".join(classification.get("search_terms", query.split()[:5]))
        
        for collection_name in collections:
            try:
                results = await retrieve_documents(
                    deps.chroma_client,
                    collection_name,
                    search_query,
                    metadata_filter if metadata_filter else None,
                    n_results=3
                )
                all_results.extend(results)
            except Exception as e:
                print(f"Error retrieving from {collection_name}: {e}")
        
        # Format results
        if not all_results:
            return "No relevant documents found."
        
        # Sort by relevance
        all_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Format context
        context = ""
        for i, doc in enumerate(all_results[:5]):  # Top 5 results
            context += f"\n[Document {i+1}]\n"
            
            if doc.get("metadata"):
                metadata = doc["metadata"]
                if metadata.get("source"):
                    context += f"Source: {metadata['source'].split('/')[-1]}\n"
                if metadata.get("subject"):
                    context += f"Subject: {metadata['subject']}\n"
            
            content = doc.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            
            context += f"Content: {content}\n"
            context += f"Relevance: {doc.get('relevance', 0):.2f}\n"
        
        return context
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from classification."""
        try:
            # Find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            print(f"JSON parsing error: {e}")
        return None
    
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
                classification["subject"] = "Mathematics AA"
        
        # Extract level
        if "hl" in query_lower or "higher level" in query_lower:
            classification["level"] = "HL"
        elif "sl" in query_lower or "standard level" in query_lower:
            classification["level"] = "SL"
        
        return classification