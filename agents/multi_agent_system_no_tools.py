"""
Multi-Agent System for IB Student Assistant

This module implements a sophisticated multi-agent architecture for handling
IB-related queries without external tool dependencies. The system orchestrates
multiple specialized AI agents to provide comprehensive responses.

Architecture:
- Fast Agent (Qwen3): Quick query classification and routing  
- RAG Agent (DeepSeek-R1): Deep reasoning with document retrieval
- Consensus Agent (DeepSeek-R1): Response review and refinement

Key Features:
- Intelligent query classification
- Multi-collection document retrieval
- IA assessment capabilities
- Response synthesis and refinement

Author: IB Student Assistant Team
Version: 2.0.0
"""

import re
import json
import chromadb
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from core.retrieval import multi_collection_search, retrieve_documents
from config.collections import DOC_TYPE_TO_COLLECTION
from config.settings import OLLAMA_HOST, MODELS
from utils.ia_assessor import IAAssessor, format_assessment_for_display


@dataclass
class MultiAgentDeps:
    """
    Dependencies for the multi-agent system.
    
    Attributes:
        chroma_client: ChromaDB client for document retrieval
        embedding_model: Model name for text embeddings (default: all-MiniLM-L6-v2)
    """
    chroma_client: chromadb.PersistentClient
    embedding_model: str = "all-MiniLM-L6-v2"


class MultiAgentSystem:
    """
    Orchestrates multiple specialized AI agents for IB-related queries.
    
    This system implements a sophisticated pipeline where different agents handle
    specific aspects of query processing:
    1. Fast Agent: Classifies and routes queries
    2. RAG Agent: Retrieves context and generates informed responses  
    3. Consensus Agent: Reviews and refines final responses
    
    Capabilities:
    - General IB program information
    - Subject-specific guidance  
    - IA assessment and feedback
    - Exam question assistance
    - Document analysis
    
    Usage:
        system = MultiAgentSystem()
        deps = MultiAgentDeps(chroma_client=client)
        result = await system.process_query("What is the IB programme?", deps)
    """
    
    def __init__(self):
        # Initialize agents with Ollama models via OpenAI compatibility
        self.fast_agent = Agent(
            OpenAIModel(
                MODELS["fast_agent"]["name"],
                provider=OpenAIProvider(
                    base_url=f"{OLLAMA_HOST}/v1",
                    api_key="ollama"
                )
            ),
            deps_type=MultiAgentDeps,
            system_prompt="""You are a query router for IB student questions. 
            Analyze the user's query and output a JSON response with:
            {
                "query_type": "general_info", // or "ia_feedback" or "exam_question" or "ia_assessment"
                "subject": "Mathematics AA", // or other subject, null if not specific
                "level": "HL", // or "SL" or null
                "search_terms": ["term1", "term2", ...] // key search terms
            }"""
        )
        
        # Use DeepSeek for RAG agent with better system prompt
        self.rag_agent = Agent(
            OpenAIModel(
                MODELS["rag_agent"]["name"],
                provider=OpenAIProvider(
                    base_url=f"{OLLAMA_HOST}/v1",
                    api_key="ollama"
                )
            ),
            deps_type=MultiAgentDeps,
            system_prompt="""You are an expert IB educator and assessor with deep knowledge of IB requirements and standards. 

When students ask about assessment criteria, IA requirements, or academic guidance:

1. **Be comprehensive and detailed** - provide thorough explanations, not brief summaries
2. **Use specific IB terminology** and reference official criteria when relevant  
3. **Structure responses clearly** with headings, bullet points, or numbered lists
4. **Include practical examples** and actionable advice
5. **Address all aspects** of the student's question completely
6. **Be encouraging yet realistic** about standards and expectations
7. **Reference specific grade boundaries or scoring** when discussing assessment

Always aim for responses that are substantive (200+ words) and genuinely helpful for student success."""
        )
        
        # Use DeepSeek for consensus agent with focus on completeness
        self.consensus_agent = Agent(
            OpenAIModel(
                MODELS["consensus_agent"]["name"],
                provider=OpenAIProvider(
                    base_url=f"{OLLAMA_HOST}/v1",
                    api_key="ollama"
                )
            ),
            deps_type=MultiAgentDeps,
            system_prompt="""You are an experienced IB coordinator reviewing student guidance. Ensure responses are:

- **Comprehensive**: Cover all aspects thoroughly, aim for 200+ words for detailed topics
- **Specific**: Include concrete details, criteria, and actionable steps  
- **Structured**: Use clear organization with headings or bullet points
- **Student-focused**: Address them directly with practical advice they can use
- **Accurate**: Reflect actual IB standards and requirements

If a response seems too brief or vague, expand it with more specific details and examples. Speak directly to the student as their knowledgeable IB advisor."""
        )
    
    def _clean_thinking_tags(self, text: str) -> Tuple[str, str]:
        """Extract thinking content and clean the main response."""
        if not text:
            return "", ""
            
        # Find all <think> content (case insensitive, multiline, greedy matching)
        thinking_pattern = r'<think>(.*?)</think>'
        thinking_matches = re.findall(thinking_pattern, text, re.DOTALL | re.IGNORECASE)
        
        # Remove ALL thinking tags and content from main response
        main_content = re.sub(thinking_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up the main content
        main_content = main_content.strip()
        
        # Remove any stray thinking tags that might be incomplete
        main_content = re.sub(r'<think.*?>', '', main_content, flags=re.IGNORECASE)
        main_content = re.sub(r'</think.*?>', '', main_content, flags=re.IGNORECASE)
        
        # Clean up excessive whitespace
        main_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', main_content)
        main_content = main_content.strip()
        
        # Join all thinking content
        thinking_content = '\n'.join(thinking_matches) if thinking_matches else ''
        thinking_content = thinking_content.strip()
        
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
                print("JSON parsing failed, using fallback classification")
                classification = self._fallback_classification(query)
                
        except Exception as e:
            print(f"Classification error: {e}")
            classification = self._fallback_classification(query)
        
        # Step 2: Retrieve context manually
        try:
            context = await self._retrieve_context(deps, query, classification)
            # If no context found, still provide comprehensive guidance
            if not context or context.strip() == "No relevant documents found.":
                context = "Based on standard IB assessment practices and criteria framework."
        except Exception as e:
            print(f"Retrieval error: {e}")
            context = "Based on standard IB assessment practices and criteria framework."
        
        # Step 3: Process with RAG agent using enhanced prompt
        try:
            prompt = f"""
As an expert IB educator, provide a comprehensive answer to this student's question.

Student's Question: {query}

Available IB Information:
{context}

Instructions for your response:
- Be thorough and detailed (aim for 200+ words for complex topics)
- Break down information clearly with structure (headings, bullets, numbers)
- Include specific examples and actionable advice
- Use proper IB terminology and reference official criteria when relevant
- Address all aspects of the question completely
- Be encouraging but maintain academic standards
- Speak directly to the student with practical guidance they can use

Provide your comprehensive response:
"""
            rag_response = await self.rag_agent.run(prompt, deps=deps)
            response_text = rag_response.data
        except Exception as e:
            print(f"RAG error: {e}")
            response_text = f"I encountered an error processing your query: {e}"
        
        # Step 4: Review and enhance for completeness
        try:
            final_prompt = f"""
Review this response to an IB student's question and ensure it's comprehensive enough.

Original Question: {query}

Current Response: {response_text}

Enhance this response to ensure it:
1. Is sufficiently detailed and comprehensive (200+ words for complex topics)
2. Addresses all aspects of the student's question
3. Includes specific, actionable guidance
4. Uses clear structure and organization
5. Provides concrete examples where helpful

Provide the enhanced response (speak directly to the student):
"""
            final_response = await self.consensus_agent.run(final_prompt, deps=deps)
            final_text = final_response.data
        except Exception as e:
            print(f"Consensus error: {e}")
            final_text = response_text
        
        # Clean thinking tags from final response
        main_response, thinking = self._clean_thinking_tags(final_text)
        
        # Fallback if main response is empty after cleaning
        if not main_response and response_text:
            main_response, thinking = self._clean_thinking_tags(response_text)
        
        # Final fallback
        if not main_response:
            main_response = "I apologize, but I encountered an issue generating a proper response. Please try rephrasing your question."
        
        return {
            "response": main_response,
            "thinking": thinking
        }
    
    async def process_ia_assessment(self, ia_text: str, subject: str, level: str, deps: MultiAgentDeps) -> Dict[str, str]:
        """Process IA assessment specifically."""
        
        try:
            # Get assessment criteria context
            context = await self._retrieve_context(
                deps, 
                f"{subject} {level} IA assessment criteria marking", 
                {"query_type": "ia_feedback", "subject": subject, "level": level}
            )
            
            # Generate assessment with RAG agent
            assessment_prompt = f"""
You are assessing an IB {subject} {level} Internal Assessment based on official IB criteria.

ASSESSMENT CRITERIA CONTEXT:
{context}

IA TEXT TO ASSESS:
{ia_text[:3000]}...

Please provide a detailed assessment including:
1. Overall score prediction and grade
2. Criterion-by-criterion feedback
3. Specific strengths identified
4. Areas needing improvement
5. Actionable next steps

Base your assessment on official IB criteria and provide constructive, detailed feedback.
"""
            
            assessment_response = await self.rag_agent.run(assessment_prompt, deps=deps)
            response_text, thinking = self._clean_thinking_tags(assessment_response.data)
            
            return {
                "response": response_text,
                "thinking": thinking,
                "classification": {"query_type": "ia_assessment", "subject": subject, "level": level}
            }
            
        except Exception as e:
            return {
                "response": f"Error assessing IA: {str(e)}",
                "thinking": f"Assessment error: {str(e)}",
                "classification": {"query_type": "error", "subject": subject, "level": level}
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
            collections = ["ia_guides", "ia_examples", "mark_schemes"]
        elif query_type == "exam_question":
            collections = ["mark_schemes", "ib_general"] 
        else:
            collections = ["ib_general", "syllabus", "ia_guides"]
        
        # Search across collections
        all_results = []
        search_query = " ".join(classification.get("search_terms", query.split()[:7]))
        
        for collection_name in collections:
            try:
                results = await retrieve_documents(
                    deps.chroma_client,
                    collection_name,
                    search_query,
                    metadata_filter if metadata_filter else None,
                    n_results=4  # Get more results for better context
                )
                all_results.extend(results)
            except Exception as e:
                print(f"Error retrieving from {collection_name}: {e}")
        
        # Format results
        if not all_results:
            return "No relevant documents found."
        
        # Sort by relevance
        all_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Format context with more detail
        context = ""
        for i, doc in enumerate(all_results[:6]):  # Use top 6 results
            context += f"\n=== Document {i+1} ===\n"
            
            if doc.get("metadata"):
                metadata = doc["metadata"]
                if metadata.get("source"):
                    context += f"Source: {metadata['source'].split('/')[-1]}\n"
                if metadata.get("subject"):
                    context += f"Subject: {metadata['subject']}\n"
                if metadata.get("doc_type"):
                    context += f"Type: {metadata['doc_type']}\n"
            
            content = doc.get("content", "")
            # Allow longer content for better context
            if len(content) > 800:
                content = content[:800] + "..."
            
            context += f"Content: {content}\n"
            context += f"Relevance: {doc.get('relevance', 0):.2f}\n"
        
        return context
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from classification."""
        try:
            # Clean the response first
            response = response.strip()
            
            # Try to find JSON object
            start = response.find('{')
            if start == -1:
                return None
                
            # Find matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(response[start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i + 1
                        break
            
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
            "search_terms": query.split()[:7]  # More search terms
        }
        
        # Detect query type
        if any(term in query_lower for term in ["ia", "internal assessment", "criteria", "marking"]):
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
