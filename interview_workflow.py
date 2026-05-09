"""
Multi-Agent Cyclic Workflow using LangGraph
Orchestrates Analyst, Interviewer, and Grader agents
"""

from langgraph.graph import StateGraph, END
from typing import Literal
from prepInterview.agents_framework import (
    WorkflowState, InterviewSession, AnalystAgent, InterviewerAgent, GraderAgent,
    Question, QuestionType, AgentFactory
)
import uuid

# ============= WORKFLOW NODES =============

def initialize_workflow(state: WorkflowState) -> WorkflowState:
    """Initialize the workflow"""
    print("\n🎯 Initializing Interview Assessment Workflow")
    print(f"   Candidate: {state['session'].candidate_name}")
    print(f"   Position: {state['session'].position_title}")
    state['current_agent'] = "analyst"
    state['action'] = "analyze"
    state['cycle_count'] = 0
    state['messages'].append(f"[SYSTEM] Workflow initialized for {state['session'].candidate_name}")
    return state

def analyst_analyze(state: WorkflowState) -> WorkflowState:
    """Analyst identifies skill gaps"""
    agent = AnalystAgent()
    session = state['session']
    
    print("\n📊 ANALYST AGENT - Analyzing Skills")
    print(f"   Resume length: {len(session.resume_text)} chars")
    print(f"   JD length: {len(session.job_description)} chars")
    
    # Reasoning phase
    reasoning = agent.reason(state, "First pass analysis")
    state['reasoning'] = reasoning
    
    # Identify gaps
    gaps = agent.identify_gaps(state)
    session.skill_gaps = gaps
    
    # Log
    agent.add_to_memory({
        "event": "Analyzed resume and job description",
        "gaps_identified": len(gaps),
        "top_gaps": [g.skill_name for g in gaps[:3]]
    })
    
    print(f"   ✓ Identified {len(gaps)} skill gaps")
    for gap in gaps[:3]:
        print(f"     • {gap.skill_name} ({gap.importance.value})")
    
    state['messages'].append(f"[ANALYST] Identified {len(gaps)} skill gaps")
    state['current_agent'] = "interviewer"
    state['action'] = "question"
    
    return state

def interviewer_generate_questions(state: WorkflowState) -> WorkflowState:
    """Interviewer generates questions based on gaps"""
    agent = InterviewerAgent()
    session = state['session']
    
    print("\n❓ INTERVIEWER AGENT - Generating Questions")
    
    # Reasoning
    reasoning = agent.reason(state, f"Cycle {state['cycle_count']}")
    state['reasoning'] = reasoning
    
    # Generate questions
    questions = agent.generate_questions(state)
    session.questions_asked.extend(questions)
    
    agent.add_to_memory({
        "event": f"Generated {len(questions)} questions",
        "cycle": state['cycle_count'],
        "questions": [q.text for q in questions]
    })
    
    print(f"   ✓ Generated {len(questions)} questions")
    for q in questions[:2]:
        print(f"     Q: {q.text[:60]}...")
    
    state['messages'].append(f"[INTERVIEWER] Generated {len(questions)} questions")
    state['current_agent'] = "candidate"
    state['action'] = "answer"
    
    return state

def candidate_answer(state: WorkflowState) -> WorkflowState:
    """Candidate answers question (simulated)"""
    session = state['session']
    
    print("\n💬 CANDIDATE ANSWERING")
    
    # Get current question
    if state['cycle_count'] == 0:
        current_q_idx = 0
    else:
        current_q_idx = session.current_question_idx
    
    if current_q_idx >= len(session.questions_asked):
        state['should_continue'] = False
        return state
    
    current_question = session.questions_asked[current_q_idx]
    
    # Simulate candidate answer (in production, get from user)
    sample_answers = [
        "I have worked with Python for 5 years, primarily focusing on backend development. I've built REST APIs using FastAPI and Flask, created data processing pipelines with Pandas and NumPy.",
        "I started learning Kubernetes last year through a personal project. I understand pods, services, and deployments, but I'm still learning about StatefulSets and operators.",
        "I haven't used Docker extensively in production, but I understand the basic concepts of containerization and have worked with docker-compose for local development.",
        "I have strong SQL skills - I'm comfortable with JOINs, aggregations, indexing strategies. I've optimized several slow queries in production databases.",
    ]
    
    answer_text = sample_answers[current_q_idx % len(sample_answers)]
    
    print(f"   Q: {current_question.text[:60]}...")
    print(f"   A: {answer_text[:60]}...")
    
    state['messages'].append(f"[CANDIDATE] Answered question {current_q_idx + 1}")
    state['current_agent'] = "grader"
    state['action'] = "grade"
    state['cycle_count'] += 1
    
    return state

def grader_evaluate(state: WorkflowState) -> WorkflowState:
    """Grader evaluates answer"""
    agent = GraderAgent()
    session = state['session']
    
    print("\n✅ GRADER AGENT - Evaluating Answer")
    
    # Get current question and simulate answer
    current_q_idx = session.current_question_idx
    if current_q_idx >= len(session.questions_asked):
        state['should_continue'] = False
        return state
    
    current_question = session.questions_asked[current_q_idx]
    
    # Simulate answer text
    sample_answers = [
        "I have 5 years of Python experience with FastAPI, Flask, and data processing. Built microservices and APIs.",
        "I learned Kubernetes recently. Understand pods, services, deployments. Still learning about advanced concepts.",
        "No production experience with Docker, but understand containerization and used docker-compose locally.",
        "Strong SQL: JOINs, aggregations, indexing. Optimized slow production queries.",
    ]
    answer_text = sample_answers[current_q_idx % len(sample_answers)]
    
    # Grade answer
    answer = agent.grade_answer(state, answer_text, current_question)
    session.answers_provided.append(answer)
    
    agent.add_to_memory({
        "event": f"Graded answer for Q{current_q_idx}",
        "clarity": answer.clarity_score,
        "accuracy": answer.technical_accuracy,
        "completeness": answer.completeness_score,
        "needs_clarification": answer.needs_clarification
    })
    
    print(f"   Clarity: {answer.clarity_score:.2f}")
    print(f"   Accuracy: {answer.technical_accuracy:.2f}")
    print(f"   Completeness: {answer.completeness_score:.2f}")
    print(f"   Overall: {answer.get_overall_score():.2f}")
    
    if answer.needs_clarification:
        print(f"   ⚠️ Needs Clarification: {answer.clarification_reason}")
    
    state['messages'].append(f"[GRADER] Score: {answer.get_overall_score():.2f}")
    
    return state

def decide_next_action(state: WorkflowState) -> Literal["clarify", "next_question", "complete"]:
    """Decide whether to clarify, ask next question, or complete"""
    session = state['session']
    
    if not session.answers_provided:
        return "next_question"
    
    last_answer = session.answers_provided[-1]
    
    # Clarification logic
    if last_answer.needs_clarification and state['cycle_count'] < 2:
        print("\n🔄 Decision: Request Clarification")
        return "clarify"
    
    # Move to next question
    if session.current_question_idx < len(session.questions_asked) - 1:
        session.current_question_idx += 1
        print("\n➡️ Decision: Move to Next Question")
        return "next_question"
    
    # Complete interview
    print("\n✔️ Decision: Complete Interview")
    return "complete"

def interviewer_clarify(state: WorkflowState) -> WorkflowState:
    """Interviewer asks clarification question"""
    agent = InterviewerAgent()
    session = state['session']
    
    print("\n🔍 INTERVIEWER - Asking Clarification")
    
    last_answer = session.answers_provided[-1]
    original_q_idx = int(last_answer.question_id.split('_')[0][1:])
    original_question = session.questions_asked[original_q_idx]
    
    # Generate clarification
    clarification_q = agent.generate_clarification(state, last_answer)
    
    if clarification_q:
        session.questions_asked.append(clarification_q)
        print(f"   Clarification Q: {clarification_q.text}")
        state['messages'].append(f"[INTERVIEWER] Asking clarification: {clarification_q.text[:60]}...")
    
    state['current_agent'] = "candidate"
    return state

def finalize_assessment(state: WorkflowState) -> WorkflowState:
    """Calculate final scores and recommendations"""
    agent = GraderAgent()
    session = state['session']
    
    print("\n📈 FINALIZING ASSESSMENT")
    
    # Calculate overall score
    overall = agent.calculate_overall_score(session)
    session.overall_score = overall
    
    print(f"   Overall Score: {overall:.2f}/1.0 ({int(overall*100)}%)")
    
    # Generate recommendations
    recommendations = []
    if overall >= 0.8:
        recommendations.append("✅ Strong candidate - Recommend for next round")
    elif overall >= 0.6:
        recommendations.append("⚠️ Average candidate - Consider further technical assessment")
    else:
        recommendations.append("❌ Candidate needs more preparation in key areas")
    
    # Gap-specific recommendations
    for gap in session.skill_gaps:
        relevant_answers = [
            a for a in session.answers_provided
            if session.questions_asked[int(a.question_id.split('_')[0][1:])].skill_gap == gap
        ]
        if relevant_answers and relevant_answers[0].get_overall_score() < 0.5:
            recommendations.append(f"🎓 Recommendation: Improve {gap.skill_name} skills")
    
    session.recommendations = recommendations
    session.interview_complete = True
    
    for rec in recommendations:
        print(f"   • {rec}")
    
    state['messages'].append("[SYSTEM] Assessment Complete")
    state['should_continue'] = False
    
    return state

# ============= LANGGRAPH WORKFLOW =============

def create_interview_workflow():
    """Create the multi-agent cyclic workflow"""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_workflow)
    workflow.add_node("analyst", analyst_analyze)
    workflow.add_node("interviewer_gen", interviewer_generate_questions)
    workflow.add_node("candidate", candidate_answer)
    workflow.add_node("grader", grader_evaluate)
    workflow.add_node("interviewer_clarify", interviewer_clarify)
    workflow.add_node("finalize", finalize_assessment)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Main flow
    workflow.add_edge("initialize", "analyst")
    workflow.add_edge("analyst", "interviewer_gen")
    workflow.add_edge("interviewer_gen", "candidate")
    workflow.add_edge("candidate", "grader")
    
    # Conditional routing after grading
    workflow.add_conditional_edges(
        "grader",
        decide_next_action,
        {
            "clarify": "interviewer_clarify",
            "next_question": "candidate",
            "complete": "finalize"
        }
    )
    
    # Clarification loop back
    workflow.add_edge("interviewer_clarify", "candidate")
    
    # Final edge
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# ============= TEST RUNNER =============

def run_interview_workflow_demo():
    """Run a complete interview workflow"""
    print("\n" + "="*70)
    print("🤖 MULTI-AGENT INTERVIEW ASSESSMENT SYSTEM")
    print("="*70)
    
    # Sample data
    resume = """
    John Doe - Senior Software Engineer
    
    Skills: Python, JavaScript, Java, SQL, Docker, Kubernetes, AWS, Django, FastAPI
    Experience: 8 years in backend development, REST APIs, microservices
    Education: BS Computer Science
    Projects: Scalable payment system, real-time analytics platform
    """
    
    job_description = """
    Senior Backend Engineer - Python/Kubernetes
    
    Required Skills:
    - 5+ years Python development
    - REST API design and implementation
    - Kubernetes orchestration
    - SQL database optimization
    - Docker containerization
    - AWS cloud platform
    - Microservices architecture
    
    Responsibilities: Design and maintain backend systems, lead technical decisions
    """
    
    # Create session
    session = InterviewSession(
        session_id=str(uuid.uuid4()),
        candidate_name="John Doe",
        position_title="Senior Backend Engineer",
        resume_text=resume,
        job_description=job_description
    )
    
    # Initialize state
    initial_state = WorkflowState(
        session=session,
        current_agent="system",
        action="initialize",
        reasoning="",
        messages=[],
        cycle_count=0,
        should_continue=True,
        error=None
    )
    
    # Create and run workflow
    workflow = create_interview_workflow()
    final_state = workflow.invoke(initial_state)
    
    # Print results
    print("\n" + "="*70)
    print("📊 ASSESSMENT RESULTS")
    print("="*70)
    print(f"Candidate: {final_state['session'].candidate_name}")
    print(f"Position: {final_state['session'].position_title}")
    print(f"Overall Score: {final_state['session'].overall_score:.2%}")
    print(f"Questions Asked: {len(final_state['session'].questions_asked)}")
    print(f"Answers Provided: {len(final_state['session'].answers_provided)}")
    
    print("\n📋 Recommendations:")
    for rec in final_state['session'].recommendations:
        print(f"   {rec}")
    
    print("\n✅ Workflow Complete!")
    
    return final_state

if __name__ == "__main__":
    result = run_interview_workflow_demo()
