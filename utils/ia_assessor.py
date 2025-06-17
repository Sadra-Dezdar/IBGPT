"""Enhanced IA assessment and feedback system for uploaded student work."""

from typing import Dict, Any, List, Optional
import re
from dataclasses import dataclass

from core.retrieval import retrieve_documents, multi_collection_search
from utils.chromadb_utils import get_chroma_client
from config.collections import DOC_TYPE_TO_COLLECTION

@dataclass
class IAAssessment:
    """Container for IA assessment results."""
    overall_score: int  # Out of 20
    criterion_scores: Dict[str, int]  # Individual criterion scores
    strengths: List[str]
    areas_for_improvement: List[str]
    specific_feedback: Dict[str, str]  # Feedback per criterion
    grade_prediction: str  # Grade band prediction
    next_steps: List[str]

class IAAssessor:
    """Comprehensive IA assessment system using IB criteria."""
    
    def __init__(self, db_dir: str = "./chroma_db"):
        """Initialize the assessor with ChromaDB access."""
        self.client = get_chroma_client(db_dir)
        
        # Math AA HL criteria (can be extended for other subjects)
        self.math_criteria = {
            "communication": {
                "weight": 4,
                "description": "Clear mathematical language, notation, and logical structure",
                "indicators": ["mathematical notation", "clear explanation", "logical flow", "appropriate terminology"]
            },
            "mathematical_presentation": {
                "weight": 4,
                "description": "Appropriate mathematical techniques and correct calculations",
                "indicators": ["correct calculations", "appropriate methods", "mathematical accuracy", "suitable techniques"]
            },
            "personal_engagement": {
                "weight": 3,
                "description": "Evidence of personal interest and authentic exploration",
                "indicators": ["personal connection", "genuine curiosity", "independent thinking", "meaningful exploration"]
            },
            "reflection": {
                "weight": 3,
                "description": "Critical evaluation of approach and consideration of limitations",
                "indicators": ["evaluates approach", "acknowledges limitations", "considers improvements", "critical thinking"]
            },
            "use_of_mathematics": {
                "weight": 6,
                "description": "Relevant mathematics applied correctly to investigate the research question",
                "indicators": ["relevant mathematics", "correct application", "sophisticated techniques", "appropriate complexity"]
            }
        }
    
    async def assess_ia(self, ia_text: str, subject: str, level: str = "HL") -> IAAssessment:
        """Assess an IA based on IB criteria and return detailed feedback."""
        
        # Get assessment criteria context from ChromaDB
        assessment_context = await self._get_assessment_context(subject, level)
        
        # Analyze IA structure and content
        structure_analysis = self._analyze_structure(ia_text)
        content_analysis = self._analyze_content(ia_text, subject)
        
        # Get criteria for the subject (defaulting to Math if not found)
        criteria = self.math_criteria if subject.lower().startswith("math") else self.math_criteria
        
        # Score each criterion
        criterion_scores = {}
        specific_feedback = {}
        
        for criterion, details in criteria.items():
            score, feedback = self._score_criterion(
                ia_text, criterion, details, assessment_context, 
                structure_analysis, content_analysis
            )
            criterion_scores[criterion] = score
            specific_feedback[criterion] = feedback
        
        # Calculate overall score
        overall_score = sum(criterion_scores.values())
        
        # Generate assessment components
        strengths = self._identify_strengths(criterion_scores, ia_text, subject)
        improvements = self._identify_improvements(criterion_scores, ia_text, subject)
        next_steps = self._generate_next_steps(criterion_scores, improvements)
        grade_prediction = self._predict_grade(overall_score, 20)
        
        return IAAssessment(
            overall_score=overall_score,
            criterion_scores=criterion_scores,
            strengths=strengths,
            areas_for_improvement=improvements,
            specific_feedback=specific_feedback,
            grade_prediction=grade_prediction,
            next_steps=next_steps
        )
        """Provide comprehensive assessment of an IA."""
        
        # Get relevant criteria and examples from database
        context = await self._get_assessment_context(subject, level)
        
        # Analyze IA structure and content
        structure_analysis = self._analyze_structure(ia_text)
        content_analysis = self._analyze_content(ia_text, subject)
        
        # Score each criterion
        criterion_scores = {}
        specific_feedback = {}
        
        if subject.lower() in ["mathematics aa", "mathematics ai"]:
            for criterion, details in self.math_criteria.items():
                score, feedback = self._score_criterion(
                    ia_text, criterion, details, context, structure_analysis, content_analysis
                )
                criterion_scores[criterion] = score
                specific_feedback[criterion] = feedback
        
        # Calculate overall score and grade
        overall_score = sum(criterion_scores.values())
        grade_prediction = self._predict_grade(overall_score, 20)  # Out of 20 for Math AA HL
        
        # Generate general feedback
        strengths = self._identify_strengths(criterion_scores, ia_text, subject)
        areas_for_improvement = self._identify_improvements(criterion_scores, ia_text, subject)
        next_steps = self._generate_next_steps(criterion_scores, areas_for_improvement)
        
        return IAAssessment(
            overall_score=overall_score,
            criterion_scores=criterion_scores,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            specific_feedback=specific_feedback,
            grade_prediction=grade_prediction,
            next_steps=next_steps
        )
    
    async def _get_assessment_context(self, subject: str, level: str) -> str:
        """Retrieve relevant assessment criteria and examples from the database."""
        # Search for IA guides
        try:
            criteria_results = await retrieve_documents(
                self.client,
                "ia_guides",
                f"{subject} IA assessment criteria {level}",
                metadata_filter={"subject": subject, "level": level},
                n_results=3
            )
            
            # Search for mark schemes
            markscheme_results = await retrieve_documents(
                self.client,
                "mark_schemes", 
                f"{subject} IA marking criteria {level}",
                metadata_filter={"subject": subject, "level": level},
                n_results=2
            )
            
            # Search for example IAs
            example_results = await retrieve_documents(
                self.client,
                "ia_examples",
                f"{subject} IA example high scoring",
                metadata_filter={"subject": subject, "level": level},
                n_results=2
            )
            
            # Combine contexts
            context = "ASSESSMENT CRITERIA:\n"
            for result in criteria_results:
                context += f"{result.get('document', '')}\n\n"
            
            context += "\nMARK SCHEMES:\n"
            for result in markscheme_results:
                context += f"{result.get('document', '')}\n\n"
            
            context += "\nEXAMPLE IAs:\n"
            for result in example_results:
                context += f"{result.get('document', '')}\n\n"
            
            return context
            
        except Exception as e:
            # Fallback to basic criteria if database query fails
            return self._get_fallback_criteria(subject, level)
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structural elements of the IA."""
        structure = {
            "word_count": len(text.split()),
            "sections": [],
            "has_introduction": False,
            "has_methodology": False,
            "has_analysis": False,
            "has_conclusion": False,
            "has_references": False,
            "mathematical_content": 0
        }
        
        # Check for key sections
        text_lower = text.lower()
        structure["has_introduction"] = any(word in text_lower for word in ["introduction", "background", "rationale"])
        structure["has_methodology"] = any(word in text_lower for word in ["methodology", "method", "approach"])
        structure["has_analysis"] = any(word in text_lower for word in ["analysis", "results", "findings"])
        structure["has_conclusion"] = any(word in text_lower for word in ["conclusion", "summary"])
        structure["has_references"] = any(word in text_lower for word in ["references", "bibliography", "sources"])
        
        # Count mathematical elements
        math_indicators = ["equation", "formula", "graph", "calculate", "derivative", "integral", "function"]
        structure["mathematical_content"] = sum(text_lower.count(indicator) for indicator in math_indicators)
        
        return structure
    
    def _analyze_content(self, text: str, subject: str) -> Dict[str, Any]:
        """Analyze the content quality and depth."""
        content = {
            "research_question_clarity": 0,
            "mathematical_sophistication": 0,
            "personal_connection": 0,
            "critical_thinking": 0,
            "use_of_sources": 0
        }
        
        text_lower = text.lower()
        
        # Research question clarity
        if "research question" in text_lower or "aim" in text_lower:
            content["research_question_clarity"] += 2
        if "?" in text and len([s for s in text.split(".") if "?" in s]) <= 3:  # Clear, focused question
            content["research_question_clarity"] += 1
        
        # Mathematical sophistication (for math subjects)
        if subject.lower() in ["mathematics aa", "mathematics ai"]:
            advanced_terms = ["calculus", "derivative", "integral", "matrix", "probability", "statistics", "modelling"]
            content["mathematical_sophistication"] = sum(text_lower.count(term) for term in advanced_terms)
        
        # Personal connection indicators
        personal_indicators = ["i chose", "my interest", "personally", "experience", "relevant to me"]
        content["personal_connection"] = sum(text_lower.count(indicator) for indicator in personal_indicators)
        
        # Critical thinking indicators
        critical_indicators = ["however", "although", "limitation", "improvement", "assumption", "validity"]
        content["critical_thinking"] = sum(text_lower.count(indicator) for indicator in critical_indicators)
        
        # Use of sources
        content["use_of_sources"] = text.count("(") + text.count("[")  # Basic citation counting
        
        return content
    
    def _score_criterion(self, text: str, criterion: str, details: Dict, context: str, 
                        structure: Dict, content: Dict) -> tuple[int, str]:
        """Score a specific criterion and provide feedback."""
        max_score = details["weight"]
        score = 0
        feedback = f"**{criterion.replace('_', ' ').title()}** ({max_score} marks):\n"
        
        if criterion == "communication":
            # Check for clear mathematical language and logical structure
            if structure["word_count"] > 2000:
                score += 1
                feedback += "✓ Adequate length for detailed explanation\n"
            
            if structure["mathematical_content"] > 5:
                score += 1
                feedback += "✓ Good use of mathematical terminology\n"
            
            if structure["has_introduction"] and structure["has_conclusion"]:
                score += 1
                feedback += "✓ Clear structure with introduction and conclusion\n"
            
            if len(text.split(".")) > 50:  # Sufficient detail
                score += 1
                feedback += "✓ Detailed explanations provided\n"
        
        elif criterion == "mathematical_presentation":
            if structure["mathematical_content"] > 10:
                score += 2
                feedback += "✓ Strong mathematical content\n"
            elif structure["mathematical_content"] > 5:
                score += 1
                feedback += "✓ Adequate mathematical content\n"
            
            # Check for calculations and methods
            if "calculate" in text.lower() or "equation" in text.lower():
                score += 1
                feedback += "✓ Evidence of calculations\n"
            
            if "graph" in text.lower() or "diagram" in text.lower():
                score += 1
                feedback += "✓ Use of visual representations\n"
        
        elif criterion == "personal_engagement":
            if content["personal_connection"] > 0:
                score += 2
                feedback += "✓ Evidence of personal interest\n"
            
            if content["research_question_clarity"] > 2:
                score += 1
                feedback += "✓ Clear, personal research question\n"
        
        elif criterion == "reflection":
            if content["critical_thinking"] > 3:
                score += 2
                feedback += "✓ Good critical evaluation\n"
            elif content["critical_thinking"] > 1:
                score += 1
                feedback += "✓ Some critical evaluation\n"
            
            if "limitation" in text.lower() or "improvement" in text.lower():
                score += 1
                feedback += "✓ Acknowledges limitations\n"
        
        elif criterion == "use_of_mathematics":
            if content["mathematical_sophistication"] > 5:
                score += 3
                feedback += "✓ Sophisticated mathematical techniques\n"
            elif content["mathematical_sophistication"] > 2:
                score += 2
                feedback += "✓ Appropriate mathematical techniques\n"
            elif structure["mathematical_content"] > 0:
                score += 1
                feedback += "✓ Basic mathematical content\n"
            
            if structure["has_analysis"]:
                score += 1
                feedback += "✓ Mathematical analysis present\n"
            
            if "model" in text.lower() or "function" in text.lower():
                score += 1
                feedback += "✓ Mathematical modeling evident\n"
            
            if structure["word_count"] > 3000:  # Detailed exploration
                score += 1
                feedback += "✓ Thorough mathematical exploration\n"
        
        # Ensure score doesn't exceed maximum
        score = min(score, max_score)
        
        if score < max_score * 0.6:  # Below 60%
            feedback += f"\n⚠️ **Needs improvement** - Score: {score}/{max_score}\n"
        elif score < max_score * 0.8:  # Below 80%
            feedback += f"\n✅ **Good work** - Score: {score}/{max_score}\n"
        else:
            feedback += f"\n🌟 **Excellent** - Score: {score}/{max_score}\n"
        
        return score, feedback
    
    def _identify_strengths(self, scores: Dict[str, int], text: str, subject: str) -> List[str]:
        """Identify the main strengths of the IA."""
        strengths = []
        
        # Check high-scoring criteria
        for criterion, score in scores.items():
            max_score = self.math_criteria[criterion]["weight"]
            if score >= max_score * 0.8:  # 80% or higher
                strengths.append(f"Strong {criterion.replace('_', ' ')}")
        
        # Content-based strengths
        if len(text.split()) > 3500:
            strengths.append("Comprehensive and detailed exploration")
        
        if "graph" in text.lower() and "equation" in text.lower():
            strengths.append("Good use of visual and mathematical representations")
        
        if text.lower().count("however") + text.lower().count("although") > 2:
            strengths.append("Demonstrates critical thinking and balanced analysis")
        
        return strengths if strengths else ["Shows understanding of the topic"]
    
    def _identify_improvements(self, scores: Dict[str, int], text: str, subject: str) -> List[str]:
        """Identify areas needing improvement."""
        improvements = []
        
        # Check low-scoring criteria
        for criterion, score in scores.items():
            max_score = self.math_criteria[criterion]["weight"]
            if score < max_score * 0.5:  # Below 50%
                improvements.append(f"Strengthen {criterion.replace('_', ' ')}")
        
        # Content-based improvements
        if len(text.split()) < 2500:
            improvements.append("Expand the exploration with more detailed analysis")
        
        if "limitation" not in text.lower():
            improvements.append("Include reflection on limitations and validity")
        
        if text.lower().count("reference") + text.lower().count("source") < 3:
            improvements.append("Include more academic sources and citations")
        
        return improvements if improvements else ["Continue developing mathematical reasoning"]
    
    def _generate_next_steps(self, scores: Dict[str, int], improvements: List[str]) -> List[str]:
        """Generate specific next steps for improvement."""
        next_steps = []
        
        # Specific action items based on low scores
        total_score = sum(scores.values())
        if total_score < 12:  # Below 60%
            next_steps.append("Focus on developing a clear research question with mathematical focus")
            next_steps.append("Add more detailed mathematical analysis and calculations")
        
        # Add steps based on improvements needed
        for improvement in improvements[:3]:  # Top 3 priorities
            if "communication" in improvement:
                next_steps.append("Improve mathematical notation and explanation clarity")
            elif "presentation" in improvement:
                next_steps.append("Include more graphs, diagrams, and mathematical work")
            elif "engagement" in improvement:
                next_steps.append("Strengthen the personal connection to your research question")
            elif "reflection" in improvement:
                next_steps.append("Add critical evaluation of your methods and findings")
            elif "mathematics" in improvement:
                next_steps.append("Incorporate more sophisticated mathematical techniques")
        
        # Always include a general next step
        next_steps.append("Review IB assessment criteria and high-scoring exemplars")
        
        return next_steps
    
    def _predict_grade(self, score: int, max_score: int) -> str:
        """Predict IB grade based on total score."""
        percentage = (score / max_score) * 100
        
        if percentage >= 87:
            return "7 (Excellent)"
        elif percentage >= 75:
            return "6 (Very Good)"
        elif percentage >= 63:
            return "5 (Good)"
        elif percentage >= 50:
            return "4 (Satisfactory)"
        elif percentage >= 37:
            return "3 (Mediocre)"
        elif percentage >= 25:
            return "2 (Poor)"
        else:
            return "1 (Very Poor)"
    
    def _get_fallback_criteria(self, subject: str, level: str) -> str:
        """Provide fallback criteria when database is unavailable."""
        return f"""
        GENERAL {subject} {level} IA ASSESSMENT CRITERIA:
        
        1. Communication (4 marks): Clear mathematical language and logical structure
        2. Mathematical Presentation (4 marks): Appropriate techniques and correct calculations  
        3. Personal Engagement (3 marks): Evidence of personal interest and exploration
        4. Reflection (3 marks): Critical evaluation and consideration of limitations
        5. Use of Mathematics (6 marks): Relevant math applied correctly to research question
        
        Total: 20 marks
        """

def format_assessment_for_display(assessment: IAAssessment) -> str:
    """Format assessment results for clear display to students."""
    
    output = f"""## 📊 IA Assessment Results

### Overall Performance
- **Total Score:** {assessment.overall_score}/20 ({(assessment.overall_score/20)*100:.0f}%)
- **Predicted Grade:** {assessment.grade_prediction}

### Criterion Breakdown
"""
    
    for criterion, score in assessment.criterion_scores.items():
        criterion_name = criterion.replace('_', ' ').title()
        output += f"- **{criterion_name}:** {score} marks\n"
    
    output += f"""
### 🌟 Strengths
"""
    for strength in assessment.strengths:
        output += f"- {strength}\n"
    
    output += f"""
### 🎯 Areas for Improvement
"""
    for improvement in assessment.areas_for_improvement:
        output += f"- {improvement}\n"
    
    output += f"""
### 📝 Detailed Feedback

"""
    for criterion, feedback in assessment.specific_feedback.items():
        output += feedback + "\n"
    
    output += f"""
### 🚀 Next Steps
"""
    for step in assessment.next_steps:
        output += f"1. {step}\n"
    
    output += """
---
*This assessment is based on IB criteria and designed to help you improve. For official grading, consult your teacher.*
"""
    
    return output
