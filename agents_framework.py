"""
Multi-Agent Interview Assessment Framework
Stateful agents that collaborate through cyclic reasoning
"""

from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from langgraph.graph import StateGraph, END
import json
from datetime import datetime

# ============= ENUMS & DATA TYPES =============

class AgentRole(Enum):
    ANALYST = "analyst"
    INTERVIEWER = "interviewer"
    GRADER = "grader"

class QuestionType(Enum):
    TECHNICAL = "technical"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"

class SkillLevel(Enum):
    CRITICAL = "critical"  # Must have
    IMPORTANT = "important"  # Nice to have
    OPTIONAL = "optional"  # Good to have

# ============= DATA CLASSES =============

@dataclass
class SkillGap:
    """Represents a skill gap between resume and job description"""
    skill_name: str
    required_level: str
    candidate_level: Optional[str]
    importance: SkillLevel
    gap_score: float  # 0.0 to 1.0
    description: str
    examples: List[str] = field(default_factory=list)

@dataclass
class Question:
    """Interview question structure"""
    id: str
    text: str
    question_type: QuestionType
    skill_gap: Optional[SkillGap] = None
    difficulty: int = 1  # 1-5 scale
    follow_up_context: str = ""
    generated_by: str = "interviewer"  # Agent name
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class Answer:
    """Candidate answer structure"""
    question_id: str
    text: str
    confidence: float  # 0.0 to 1.0 (from grader)
    clarity_score: float  # 0.0 to 1.0
    technical_accuracy: float  # 0.0 to 1.0
    completeness_score: float  # 0.0 to 1.0
    needs_clarification: bool = False
    clarification_reason: str = ""
    graded_by: str = "grader"
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def get_overall_score(self) -> float:
        """Calculate weighted overall score"""
        return (
            self.clarity_score * 0.25 +
            self.technical_accuracy * 0.4 +
            self.completeness_score * 0.25 +
            self.confidence * 0.1
        )

@dataclass
class InterviewSession:
    """Complete interview session state"""
    session_id: str
    candidate_name: str
    position_title: str
    resume_text: str
    job_description: str
    skill_gaps: List[SkillGap] = field(default_factory=list)
    questions_asked: List[Question] = field(default_factory=list)
    answers_provided: List[Answer] = field(default_factory=list)
    current_question_idx: int = 0
    clarification_round: int = 0
    interview_complete: bool = False
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    agent_conversation_log: List[Dict[str, str]] = field(default_factory=list)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

# ============= LANGGRAPH STATE =============

class WorkflowState(TypedDict):
    """Shared state for all agents"""
    session: InterviewSession
    current_agent: str  # "analyst", "interviewer", "grader"
    action: str  # "analyze", "question", "grade", "clarify", "conclude"
    reasoning: str  # Agent's reasoning
    messages: List[str]  # Conversation history
    cycle_count: int  # Track cyclic interactions
    should_continue: bool  # Continue interview?
    error: Optional[str]  # Error tracking

# ============= AGENT BASE CLASS =============

class BaseAgent:
    """Base class for all interview agents"""
    
    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.memory: List[Dict[str, Any]] = []
    
    def add_to_memory(self, event: Dict[str, Any]):
        """Store interaction in memory"""
        event['timestamp'] = datetime.now().isoformat()
        event['agent'] = self.name
        self.memory.append(event)
    
    def get_memory_summary(self, last_n: int = 10) -> str:
        """Get summary of recent interactions"""
        recent = self.memory[-last_n:] if self.memory else []
        summary = f"\n{self.name}'s Memory:\n"
        for item in recent:
            summary += f"  • {item.get('event', '')}\n"
        return summary
    
    def reason(self, state: WorkflowState, context: str) -> str:
        """Agent reasoning function"""
        raise NotImplementedError


# ============= ANALYST AGENT =============

class AnalystAgent(BaseAgent):
    """
    Analyzes resume and job description to identify skill gaps
    """
    
    def __init__(self):
        super().__init__("Analyst", AgentRole.ANALYST)
        self.skill_extraction_patterns = {
            "programming": ["python", "java", "javascript", "c++", "rust"],
            "data": ["sql", "pandas", "numpy", "spark", "hadoop"],
            "cloud": ["aws", "gcp", "azure", "kubernetes", "docker"],
            "ml": ["tensorflow", "pytorch", "scikit-learn", "nlp", "cv"],
            "backend": ["rest api", "microservices", "spring", "django", "fastapi"],
            "frontend": ["react", "vue", "angular", "typescript", "css"],
        }
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text"""
        skills_found = {}
        text_lower = text.lower()
        
        for category, skills in self.skill_extraction_patterns.items():
            found = [s for s in skills if s in text_lower]
            if found:
                skills_found[category] = found
        
        return skills_found
    
    def identify_gaps(self, state: WorkflowState) -> List[SkillGap]:
        """Identify skill gaps between resume and job description"""
        session = state['session']
        
        # Extract skills
        resume_skills = self.extract_skills(session.resume_text)
        jd_skills = self.extract_skills(session.job_description)
        
        gaps = []
        gap_id = 0
        
        # Find missing skills in resume
        for category, required_skills in jd_skills.items():
            candidate_skills = resume_skills.get(category, [])
            
            for skill in required_skills:
                if skill not in candidate_skills:
                    importance = self._determine_importance(skill, session.job_description)
                    gap_score = 1.0 if not candidate_skills else 0.7
                    
                    gap = SkillGap(
                        skill_name=skill,
                        required_level="Intermediate",
                        candidate_level=None,
                        importance=importance,
                        gap_score=gap_score,
                        description=f"Skill '{skill}' mentioned in job description but not found in resume",
                        examples=self._get_skill_examples(skill)
                    )
                    gaps.append(gap)
                    gap_id += 1
        
        return gaps[:5]  # Top 5 gaps
    
    def _determine_importance(self, skill: str, job_description: str) -> SkillLevel:
        """Determine importance based on mentions in JD"""
        jd_lower = job_description.lower()
        skill_lower = skill.lower()
        
        # Count mentions
        mention_count = jd_lower.count(skill_lower)
        
        if mention_count >= 3:
            return SkillLevel.CRITICAL
        elif mention_count == 2:
            return SkillLevel.IMPORTANT
        else:
            return SkillLevel.OPTIONAL
    
    def _get_skill_examples(self, skill: str) -> List[str]:
        """Get example scenarios for a skill"""
        examples = {
            "python": [
                "Write a function to find duplicate items in a list",
                "Explain decorators and their use cases",
                "How would you optimize a slow Python script?"
            ],
            "rest api": [
                "Design a REST API for a todo application",
                "Explain the difference between PUT and PATCH",
                "How do you handle versioning in APIs?"
            ],
            "sql": [
                "Write a query to find users with multiple orders",
                "Explain the difference between JOIN types",
                "How would you optimize slow queries?"
            ],
            "docker": [
                "What is the difference between containers and VMs?",
                "How would you containerize a Python application?",
                "Explain Docker Compose and when to use it"
            ],
            "kubernetes": [
                "What are the main components of Kubernetes?",
                "How do you handle scaling in Kubernetes?",
                "Explain deployments and services"
            ]
        }
        return examples.get(skill.lower(), [
            f"Explain your experience with {skill}",
            f"Describe a project where you used {skill}",
            f"What challenges did you face with {skill}?"
        ])
    
    def reason(self, state: WorkflowState, context: str) -> str:
        """Analyst reasoning"""
        session = state['session']
        return f"""
        Analyzing candidate profile for {session.position_title} position.
        
        Resume Summary: {len(session.resume_text)} characters
        Job Description Summary: {len(session.job_description)} characters
        
        Context: {context}
        
        Analysis Goal: Identify critical skill gaps to focus interview questions.
        Strategy: Extract technical skills from both documents and compare.
        """


# ============= INTERVIEWER AGENT =============

class InterviewerAgent(BaseAgent):
    """
    Generates interview questions based on skill gaps
    Uses cyclic feedback to ask clarification questions
    """
    
    def __init__(self):
        super().__init__("Interviewer", AgentRole.INTERVIEWER)
        self.question_templates = {
            1: "Can you explain your experience with {skill}?",
            2: "Tell me about a project where you used {skill}.",
            3: "What challenges did you face with {skill} and how did you solve them?",
            4: "How would you approach {skill} in a production environment?",
            5: "Compare {skill} with alternatives. Why would you choose {skill}?"
        }
    
    def generate_questions(self, state: WorkflowState) -> List[Question]:
        """Generate questions based on skill gaps"""
        session = state['session']
        questions = []
        
        for idx, gap in enumerate(session.skill_gaps[:3]):
            difficulty = 2 if gap.importance == SkillLevel.CRITICAL else 1
            
            # Select template based on importance
            template_num = min(difficulty + 2, 5)
            template = self.question_templates.get(template_num, self.question_templates[1])
            
            question_text = template.format(skill=gap.skill_name)
            
            question = Question(
                id=f"q_{len(session.questions_asked) + idx}",
                text=question_text,
                question_type=QuestionType.TECHNICAL,
                skill_gap=gap,
                difficulty=difficulty,
                generated_by=self.name
            )
            questions.append(question)
        
        return questions
    
    def generate_clarification(self, state: WorkflowState, answer: Answer) -> Optional[Question]:
        """Generate clarification question if answer was vague"""
        if not answer.needs_clarification:
            return None
        
        session = state['session']
        original_question = session.questions_asked[
            int(answer.question_id.split('_')[1])
        ]
        
        clarification_templates = [
            "You mentioned {context}. Can you provide a specific example?",
            "How exactly would you implement {context}?",
            "What tools or frameworks would you use for {context}?",
            "Walk me through your thought process for {context}.",
        ]
        
        template = clarification_templates[state['cycle_count'] % len(clarification_templates)]
        
        question = Question(
            id=f"q_{len(session.questions_asked)}_clarify",
            text=template.format(context=answer.clarification_reason),
            question_type=QuestionType.CLARIFICATION,
            skill_gap=original_question.skill_gap,
            difficulty=original_question.difficulty,
            follow_up_context=original_question.text,
            generated_by=self.name
        )
        
        return question
    
    def reason(self, state: WorkflowState, context: str) -> str:
        """Interviewer reasoning"""
        session = state['session']
        gap_skills = [g.skill_name for g in session.skill_gaps]
        
        return f"""
        Generating interview questions to assess candidate on identified gaps.
        
        Gap Skills to Focus: {', '.join(gap_skills)}
        Cycle Count: {state['cycle_count']}
        
        Context: {context}
        
        Question Strategy: 
        1. Start with technical questions on critical gaps
        2. If answers are vague, request clarification
        3. Progressively increase difficulty
        """


# ============= GRADER AGENT =============

class GraderAgent(BaseAgent):
    """
    Evaluates candidate answers
    Determines if clarification is needed or if we should move forward
    """
    
    def __init__(self):
        super().__init__("Grader", AgentRole.GRADER)
    
    def grade_answer(self, state: WorkflowState, answer_text: str, question: Question) -> Answer:
        """
        Grade candidate answer
        Returns Answer object with clarity, accuracy, completeness scores
        """
        # Simulate grading (in production, use LLM)
        clarity = self._calculate_clarity(answer_text)
        accuracy = self._calculate_accuracy(answer_text, question)
        completeness = self._calculate_completeness(answer_text, question)
        confidence = (clarity + accuracy) / 2
        
        needs_clarification = clarity < 0.6 or completeness < 0.5
        
        session = state['session']
        answer = Answer(
            question_id=question.id,
            text=answer_text,
            confidence=confidence,
            clarity_score=clarity,
            technical_accuracy=accuracy,
            completeness_score=completeness,
            needs_clarification=needs_clarification,
            clarification_reason=self._get_clarification_reason(
                clarity, accuracy, completeness
            ),
            graded_by=self.name
        )
        
        return answer
    
    def _calculate_clarity(self, text: str) -> float:
        """Estimate answer clarity (0.0 to 1.0)"""
        # Simple heuristic: longer, structured answers are clearer
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Optimal: 10-20 words per sentence
        if 10 <= avg_sentence_length <= 20:
            return 0.9
        elif 5 <= avg_sentence_length <= 30:
            return 0.7
        else:
            return 0.4
    
    def _calculate_accuracy(self, text: str, question: Question) -> float:
        """Estimate technical accuracy based on keywords"""
        if not question.skill_gap:
            return 0.5
        
        skill_keywords = self._get_keywords_for_skill(question.skill_gap.skill_name)
        text_lower = text.lower()
        
        matched_keywords = sum(1 for kw in skill_keywords if kw in text_lower)
        accuracy = min(matched_keywords / len(skill_keywords), 1.0) if skill_keywords else 0.5
        
        return accuracy
    
    def _calculate_completeness(self, text: str, question: Question) -> float:
        """Estimate answer completeness"""
        word_count = len(text.split())
        
        if word_count < 20:
            return 0.3  # Too short
        elif word_count < 50:
            return 0.6  # Decent
        else:
            return 0.9  # Comprehensive
    
    def _get_keywords_for_skill(self, skill: str) -> List[str]:
        """Get keywords associated with a skill"""
        keywords = {
            "python": ["function", "class", "module", "decorator", "comprehension"],
            "rest api": ["endpoint", "status code", "request", "response", "json"],
            "sql": ["query", "join", "index", "aggregate", "optimize"],
            "docker": ["container", "image", "compose", "volume", "port"],
            "kubernetes": ["pod", "deployment", "service", "replica", "scale"],
        }
        return keywords.get(skill.lower(), ["experience", "project", "implement"])
    
    def _get_clarification_reason(self, clarity: float, accuracy: float, completeness: float) -> str:
        """Determine why clarification is needed"""
        if clarity < 0.6:
            return "Answer was unclear. Please provide more specific details."
        elif accuracy < 0.5:
            return "Technical details seem incomplete. Can you elaborate?"
        elif completeness < 0.5:
            return "Answer was too brief. Please provide a more complete response."
        return ""
    
    def calculate_overall_score(self, session: InterviewSession) -> float:
        """Calculate overall interview score"""
        if not session.answers_provided:
            return 0.0
        
        scores = [answer.get_overall_score() for answer in session.answers_provided]
        return sum(scores) / len(scores)
    
    def reason(self, state: WorkflowState, context: str) -> str:
        """Grader reasoning"""
        session = state['session']
        
        return f"""
        Evaluating candidate response against question criteria.
        
        Criteria:
        - Clarity: Is the answer clear and understandable?
        - Accuracy: Does the answer demonstrate correct technical knowledge?
        - Completeness: Does the answer fully address the question?
        
        Context: {context}
        
        Decision Matrix:
        - If all scores > 0.7: Move to next question
        - If any score < 0.6: Request clarification
        - If pattern of low scores: Reassess candidate fit
        """


# ============= AGENT FACTORY =============

class AgentFactory:
    """Factory for creating interview agents"""
    
    _agents = {
        "analyst": AnalystAgent,
        "interviewer": InterviewerAgent,
        "grader": GraderAgent
    }
    
    @classmethod
    def create_agent(cls, role: str) -> BaseAgent:
        """Create an agent by role"""
        agent_class = cls._agents.get(role.lower())
        if not agent_class:
            raise ValueError(f"Unknown agent role: {role}")
        return agent_class()
    
    @classmethod
    def create_all_agents(cls) -> Dict[str, BaseAgent]:
        """Create all agents"""
        return {
            "analyst": cls.create_agent("analyst"),
            "interviewer": cls.create_agent("interviewer"),
            "grader": cls.create_agent("grader")
        }


print("✅ Multi-Agent Framework Loaded!")
print("Agents: Analyst, Interviewer, Grader")
