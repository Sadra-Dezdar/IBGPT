"""Exam question handler tool for the RAG system."""

from typing import Dict, Any
from pydantic_ai import RunContext, Tool

from tools.context_retriever import retrieve_context

@Tool
async def solve_exam_question(
    context: RunContext,
    question: str,
    subject: str,
    level: str = "HL",
    paper: str = None
) -> str:
    """Solve an exam question using past papers and mark schemes."""
    
    # Retrieve relevant past paper questions
    paper_query = f"{subject} {level} exam question similar to: {question}"
    
    past_paper_context = await retrieve_context(
        context,
        paper_query,
        doc_type="general_info",  # Since we removed past_paper type
        subject=subject,
        level=level,
        n_results=3
    )
    
    # Retrieve relevant mark schemes
    mark_scheme_query = f"{subject} {level} mark scheme {question[:50]}"
    
    mark_scheme_context = await retrieve_context(
        context,
        mark_scheme_query,
        doc_type="mark_scheme",
        subject=subject,
        level=level,
        n_results=3
    )
    
    # Retrieve formula booklet if math
    formula_context = ""
    if "math" in subject.lower():
        formula_query = "mathematics formula booklet"
        formula_context = await retrieve_context(
            context,
            formula_query,
            doc_type="general_info",
            n_results=2
        )
    
    # Combine contexts
    full_context = f"""
SIMILAR PAST PAPER QUESTIONS:
{past_paper_context}

RELEVANT MARK SCHEMES:
{mark_scheme_context}
"""
    
    if formula_context:
        full_context += f"\nFORMULA BOOKLET REFERENCES:\n{formula_context}"
    
    full_context += f"\nQUESTION TO SOLVE:\n{question}"
    
    # Return the context for the agent to process
    return f"""
Based on IB past papers and mark schemes, please solve this exam question.

{full_context}

Please structure your answer following IB mark scheme format:
1. Show clear working/reasoning
2. State the method being used
3. Present calculations step by step
4. Box or clearly state the final answer
5. Indicate mark allocation for each step
"""