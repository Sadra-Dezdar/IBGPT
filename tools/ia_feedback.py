"""IA feedback tool for the RAG system."""

from typing import Dict, Any
from pydantic_ai import RunContext, Tool

from core.retrieval import retrieve_documents
from tools.context_retriever import retrieve_context

@Tool 
async def provide_ia_feedback(
    context: RunContext,
    ia_text: str,
    subject: str,
    level: str = "HL"
) -> str:
    """Provide feedback on an IA based on IB criteria."""
    
    # First, retrieve IA assessment criteria for the subject
    criteria_query = f"{subject} IA assessment criteria {level}"
    
    criteria_context = await retrieve_context(
        context,
        criteria_query,
        doc_type="ia_guide",
        subject=subject,
        level=level,
        n_results=3
    )
    
    # Then, retrieve example IAs for comparison
    example_query = f"{subject} IA example high score"
    
    example_context = await retrieve_context(
        context,
        example_query,
        doc_type="ia_example",
        subject=subject,
        level=level,
        n_results=2
    )
    
    # Combine contexts
    full_context = f"""
ASSESSMENT CRITERIA:
{criteria_context}

EXAMPLE IAS:
{example_context}

STUDENT IA TO ASSESS:
{ia_text[:1000]}...  # Truncate if too long
"""
    
    # Return the context for the agent to process
    return f"""
Based on the IB assessment criteria and example IAs, please provide feedback on this student's IA.

{full_context}

Please structure your feedback by:
1. Identifying strengths according to IB criteria
2. Suggesting areas for improvement
3. Estimating a score range based on the criteria
4. Providing specific recommendations for enhancement
"""