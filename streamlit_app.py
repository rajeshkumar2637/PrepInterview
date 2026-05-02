"""
Streamlit Dashboard for Multi-Agent Interview Assessment
Beautiful, interactive UI for the interview workflow
"""

import streamlit as st
import json
from datetime import datetime
from prepInterview.interview_workflow import create_interview_workflow
from prepInterview.agents_framework import WorkflowState, InterviewSession
import uuid
import os

# PAGE CONFIGURATION

st.set_page_config(
    page_title="🤖 AI Interview Assessment",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  CUSTOM CSS

st.markdown("""
<style>
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .stMetric {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .score-excellent {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .score-good {
        color: #17a2b8;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .score-fair {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .score-poor {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .question-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .answer-card {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .skill-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============= INITIALIZATION =============

if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = None

if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False

if 'current_step' not in st.session_state:
    st.session_state.current_step = "upload"

# ============= HEADER =============

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🤖 AI Interview Assessment System")
    st.markdown("#### Multi-Agent Workflow for Smart Candidate Evaluation")

st.markdown("---")

#  SIDEBAR 

with st.sidebar:
    st.header("📋 Navigation")
    
    page = st.radio(
        "Select View",
        ["📤 Upload & Configure", "🎯 Live Assessment", "📊 Results Dashboard", "⚙️ Settings"]
    )
    
    st.markdown("---")
    st.header("📊 Quick Stats")
    
    if st.session_state.workflow_state:
        session = st.session_state.workflow_state['session']
        st.metric("Questions", len(session.questions_asked))
        st.metric("Answers", len(session.answers_provided))
        st.metric("Score", f"{session.overall_score:.1%}")
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    **Multi-Agent System:**
    - 🔍 **Analyst**: Identifies skill gaps
    - ❓ **Interviewer**: Generates questions
    - ✅ **Grader**: Evaluates answers
    
    **Tech Stack:**
    - LangGraph for workflow
    - ChromaDB for vector storage
    - FastAPI backend
    - Streamlit frontend
    """)

# ============= MAIN CONTENT

if page == "📤 Upload & Configure":
    st.header("📤 Upload Resume & Job Description")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Candidate Information")
        candidate_name = st.text_input("Candidate Name", "John Doe")
        position_title = st.text_input("Position Title", "Senior Backend Engineer")
    
    with col2:
        st.subheader("📝 Upload Documents")
        resume_file = st.file_uploader("Upload Resume (TXT/PDF)", type=["txt", "pdf"])
        jd_file = st.file_uploader("Upload Job Description (TXT/PDF)", type=["txt", "pdf"])
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Resume Text")
        resume_text = st.text_area(
            "Or paste resume here:",
            height=200,
            value="""John Doe - Senior Software Engineer

Skills: Python, JavaScript, Java, SQL, Docker, Kubernetes, AWS, Django, FastAPI
Experience: 8 years in backend development, REST APIs, microservices
Education: BS Computer Science
Projects: Scalable payment system, real-time analytics platform"""
        )
    
    with col2:
        st.subheader("📋 Job Description Text")
        jd_text = st.text_area(
            "Or paste job description here:",
            height=200,
            value="""Senior Backend Engineer - Python/Kubernetes

Required Skills:
- 5+ years Python development
- REST API design and implementation
- Kubernetes orchestration
- SQL database optimization
- Docker containerization
- AWS cloud platform
- Microservices architecture"""
        )
    
    st.markdown("---")
    
    st.subheader("⚙️ Interview Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_questions = st.slider("Number of Questions", 3, 10, 5)
    
    with col2:
        difficulty = st.select_slider("Difficulty Level", ["Easy", "Medium", "Hard"])
    
    with col3:
        focus_areas = st.multiselect(
            "Focus Areas",
            ["Technical Skills", "Problem Solving", "Communication", "Leadership"],
            default=["Technical Skills"]
        )
    
    st.markdown("---")
    
    if st.button("🚀 Start Interview Assessment", use_container_width=True):
        # Create interview session
        session = InterviewSession(
            session_id=str(uuid.uuid4()),
            candidate_name=candidate_name,
            position_title=position_title,
            resume_text=resume_text,
            job_description=jd_text
        )
        
        # Initialize workflow state
        workflow_state = WorkflowState(
            session=session,
            current_agent="system",
            action="initialize",
            reasoning="",
            messages=[],
            cycle_count=0,
            should_continue=True,
            error=None
        )
        
        st.session_state.workflow_state = workflow_state
        st.session_state.current_step = "assessment"
        st.success("✅ Interview session created! Redirecting...")
        st.rerun()

elif page == "🎯 Live Assessment":
    st.header("🎯 Live Interview Assessment")
    
    if not st.session_state.workflow_state:
        st.warning("⚠️ Please upload documents and configure interview first")
        st.info("Go to 'Upload & Configure' tab to get started")
    else:
        session = st.session_state.workflow_state['session']
        
        # Header info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Candidate", session.candidate_name)
        with col2:
            st.metric("Position", session.position_title)
        with col3:
            st.metric("Status", "In Progress" if not session.interview_complete else "Complete")
        
        st.markdown("---")
        
        # Workflow execution
        if st.button("▶️ Run Workflow", use_container_width=True):
            with st.spinner("🔄 Running multi-agent workflow..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create workflow
                workflow = create_interview_workflow()
                
                # Run workflow
                final_state = workflow.invoke(st.session_state.workflow_state)
                st.session_state.workflow_state = final_state
                
                progress_bar.progress(100)
                status_text.success("✅ Workflow Complete!")
                st.session_state.interview_complete = True
        
        st.markdown("---")
        
        # Display current state
        st.subheader("📋 Questions & Answers")
        
        if session.questions_asked:
            for idx, question in enumerate(session.questions_asked[:5], 1):
                with st.expander(f"Q{idx}: {question.text[:60]}..."):
                    st.write(f"**Type**: {question.question_type.value}")
                    st.write(f"**Difficulty**: {'⭐' * question.difficulty}")
                    st.write(f"**Full Question**: {question.text}")
                    
                    if idx <= len(session.answers_provided):
                        answer = session.answers_provided[idx-1]
                        st.markdown("**Answer:**")
                        st.write(answer.text)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Clarity", f"{answer.clarity_score:.1%}")
                        with col2:
                            st.metric("Accuracy", f"{answer.technical_accuracy:.1%}")
                        with col3:
                            st.metric("Complete", f"{answer.completeness_score:.1%}")
                        with col4:
                            st.metric("Overall", f"{answer.get_overall_score():.1%}")
                        
                        if answer.needs_clarification:
                            st.warning(f"⚠️ Needs clarification: {answer.clarification_reason}")

elif page == "📊 Results Dashboard":
    st.header("📊 Assessment Results Dashboard")
    
    if not st.session_state.workflow_state or not st.session_state.interview_complete:
        st.warning("⚠️ No completed assessment yet")
        st.info("Go to 'Live Assessment' to run the workflow first")
    else:
        session = st.session_state.workflow_state['session']
        
        # Overall Score
        col1, col2, col3 = st.columns(3)
        
        score = session.overall_score
        with col1:
            if score >= 0.8:
                st.markdown(f'<p class="score-excellent">Overall Score: {score:.1%}</p>', 
                           unsafe_allow_html=True)
            elif score >= 0.6:
                st.markdown(f'<p class="score-good">Overall Score: {score:.1%}</p>', 
                           unsafe_allow_html=True)
            elif score >= 0.4:
                st.markdown(f'<p class="score-fair">Overall Score: {score:.1%}</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="score-poor">Overall Score: {score:.1%}</p>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.metric("Questions Asked", len(session.questions_asked))
        
        with col3:
            st.metric("Questions Answered", len(session.answers_provided))
        
        st.markdown("---")
        
        # Skills Analysis
        st.subheader("🔍 Skills Gap Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Identified Gaps:**")
            for gap in session.skill_gaps:
                st.markdown(f'<span class="skill-badge">{gap.skill_name}</span>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.write("**Importance Levels:**")
            for gap in session.skill_gaps:
                st.write(f"• {gap.skill_name}: {gap.importance.value}")
        
        st.markdown("---")
        
        # Answer Quality Breakdown
        st.subheader("📈 Answer Quality Analysis")
        
        if session.answers_provided:
            scores_data = {
                "Clarity": [a.clarity_score for a in session.answers_provided],
                "Accuracy": [a.technical_accuracy for a in session.answers_provided],
                "Completeness": [a.completeness_score for a in session.answers_provided],
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_clarity = sum(scores_data["Clarity"]) / len(scores_data["Clarity"])
                st.metric("Avg Clarity", f"{avg_clarity:.1%}")
            
            with col2:
                avg_accuracy = sum(scores_data["Accuracy"]) / len(scores_data["Accuracy"])
                st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
            
            with col3:
                avg_complete = sum(scores_data["Completeness"]) / len(scores_data["Completeness"])
                st.metric("Avg Completeness", f"{avg_complete:.1%}")
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if session.recommendations:
            for i, rec in enumerate(session.recommendations, 1):
                if "Strong" in rec or "✅" in rec:
                    st.success(rec)
                elif "Average" in rec or "⚠️" in rec:
                    st.warning(rec)
                else:
                    st.info(rec)
        
        st.markdown("---")
        
        # Export Results
        st.subheader("📥 Export Results")
        
        if st.button("📊 Download Results as JSON"):
            results = {
                "session_id": session.session_id,
                "candidate": session.candidate_name,
                "position": session.position_title,
                "overall_score": session.overall_score,
                "questions": len(session.questions_asked),
                "answers": len(session.answers_provided),
                "recommendations": session.recommendations,
                "created_at": session.created_at
            }
            
            st.json(results)

elif page == "⚙️ Settings":
    st.header("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Agent Configuration")
        
        analyst_enabled = st.checkbox("Enable Analyst Agent", True)
        interviewer_enabled = st.checkbox("Enable Interviewer Agent", True)
        grader_enabled = st.checkbox("Enable Grader Agent", True)
        
        st.markdown("---")
        
        st.subheader("📝 Grading Criteria")
        clarity_weight = st.slider("Clarity Weight", 0.0, 1.0, 0.25)
        accuracy_weight = st.slider("Accuracy Weight", 0.0, 1.0, 0.4)
        completeness_weight = st.slider("Completeness Weight", 0.0, 1.0, 0.25)
        confidence_weight = st.slider("Confidence Weight", 0.0, 1.0, 0.1)
    
    with col2:
        st.subheader("🗄️ Storage Configuration")
        
        chromadb_enabled = st.checkbox("Enable ChromaDB", True)
        chromadb_collection = st.text_input("Collection Name", "interviews")
        
        st.markdown("---")
        
        st.subheader("🚀 Deployment")
        
        deployment_type = st.selectbox(
            "Deployment Type",
            ["Local", "Docker", "Kubernetes", "Cloud (AWS/GCP)"]
        )
        
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved!")

# FOOTER 

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Built with** 🤖 LangGraph + 🎯 Streamlit")
with col2:
    st.markdown("**Powered by** 🐍 Python + 🚀 FastAPI")
with col3:
    st.markdown("**Version** 1.0.0 | 2026")

print("✅ Streamlit App Ready!")
