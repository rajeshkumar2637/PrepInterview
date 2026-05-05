"""
FastAPI Backend for Multi-Agent Interview Assessment
REST API endpoints for workflow execution, storage, and management
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
from prepInterview.interview_workflow import create_interview_workflow
from prepInterview.agents_framework import WorkflowState, InterviewSession
import asyncio

# ============= FASTAPI APP =============

app = FastAPI(
    title="🤖 Multi-Agent Interview API",
    description="REST API for AI-powered interview assessment",
    version="1.0.0"
)

# ============= CORS CONFIGURATION =============

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= REQUEST/RESPONSE MODELS =============

class InterviewCreateRequest(BaseModel):
    candidate_name: str
    position_title: str
    resume_text: str
    job_description: str
    num_questions: int = 5
    difficulty: str = "medium"

class InterviewResponse(BaseModel):
    session_id: str
    candidate_name: str
    position_title: str
    status: str
    score: float = 0.0
    questions_count: int
    answers_count: int
    created_at: str

class AnswerSubmitRequest(BaseModel):
    session_id: str
    question_id: str
    answer_text: str

class WorkflowExecuteRequest(BaseModel):
    session_id: str
    resume_text: str
    job_description: str

class SkillGapResponse(BaseModel):
    skill_name: str
    required_level: str
    importance: str
    gap_score: float
    description: str

class QuestionResponse(BaseModel):
    id: str
    text: str
    question_type: str
    difficulty: int
    skill_gap: Optional[Dict[str, Any]] = None

class AnswerScoreResponse(BaseModel):
    question_id: str
    clarity_score: float
    accuracy_score: float
    completeness_score: float
    overall_score: float
    needs_clarification: bool
    clarification_reason: str = ""

class AssessmentResultResponse(BaseModel):
    session_id: str
    candidate_name: str
    position_title: str
    overall_score: float
    questions_count: int
    answers_count: int
    skill_gaps: List[SkillGapResponse]
    recommendations: List[str]
    completed_at: str

# ============= IN-MEMORY STORAGE (Replace with DB for production) =============

sessions_db: Dict[str, Dict[str, Any]] = {}
workflow_jobs: Dict[str, Dict[str, Any]] = {}

# ============= HEALTH CHECK =============

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# ============= INTERVIEW MANAGEMENT ENDPOINTS =============

@app.post("/api/interviews/create", response_model=InterviewResponse, tags=["Interviews"])
async def create_interview(request: InterviewCreateRequest):
    """Create new interview session"""
    session_id = str(uuid.uuid4())
    
    session = InterviewSession(
        session_id=session_id,
        candidate_name=request.candidate_name,
        position_title=request.position_title,
        resume_text=request.resume_text,
        job_description=request.job_description
    )
    
    sessions_db[session_id] = {
        "session": session,
        "state": None,
        "status": "created"
    }
    
    return InterviewResponse(
        session_id=session_id,
        candidate_name=session.candidate_name,
        position_title=session.position_title,
        status="created",
        questions_count=0,
        answers_count=0,
        created_at=session.created_at
    )

@app.get("/api/interviews/{session_id}", response_model=InterviewResponse, tags=["Interviews"])
async def get_interview(session_id: str):
    """Get interview details"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    db_entry = sessions_db[session_id]
    session = db_entry["session"]
    
    return InterviewResponse(
        session_id=session.session_id,
        candidate_name=session.candidate_name,
        position_title=session.position_title,
        status=db_entry["status"],
        score=session.overall_score,
        questions_count=len(session.questions_asked),
        answers_count=len(session.answers_provided),
        created_at=session.created_at
    )

@app.get("/api/interviews", tags=["Interviews"])
async def list_interviews(skip: int = 0, limit: int = 10):
    """List all interview sessions"""
    interviews = []
    for session_id, db_entry in list(sessions_db.items())[skip:skip+limit]:
        session = db_entry["session"]
        interviews.append({
            "session_id": session.session_id,
            "candidate_name": session.candidate_name,
            "position_title": session.position_title,
            "status": db_entry["status"],
            "score": session.overall_score,
            "created_at": session.created_at
        })
    
    return {
        "total": len(sessions_db),
        "items": interviews
    }

# ============= WORKFLOW EXECUTION ENDPOINTS =============

@app.post("/api/workflows/execute", tags=["Workflows"])
async def execute_workflow(request: WorkflowExecuteRequest, background_tasks: BackgroundTasks):
    """Execute the multi-agent interview workflow"""
    if request.session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_entry = sessions_db[request.session_id]
    session = db_entry["session"]
    
    # Update session with documents
    session.resume_text = request.resume_text
    session.job_description = request.job_description
    
    # Create workflow state
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
    
    # Create job tracking
    job_id = str(uuid.uuid4())
    workflow_jobs[job_id] = {
        "session_id": request.session_id,
        "status": "running",
        "progress": 0.0,
        "started_at": datetime.now().isoformat()
    }
    
    # Run workflow in background
    async def run_workflow():
        try:
            workflow = create_interview_workflow()
            final_state = workflow.invoke(workflow_state)
            
            db_entry["state"] = final_state
            db_entry["status"] = "completed"
            
            workflow_jobs[job_id]["status"] = "completed"
            workflow_jobs[job_id]["progress"] = 1.0
            workflow_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            workflow_jobs[job_id]["status"] = "failed"
            workflow_jobs[job_id]["error"] = str(e)
    
    background_tasks.add_task(run_workflow)
    
    return {
        "job_id": job_id,
        "session_id": request.session_id,
        "status": "queued",
        "message": "Workflow execution started in background"
    }

@app.get("/api/workflows/jobs/{job_id}", tags=["Workflows"])
async def get_job_status(job_id: str):
    """Get workflow job status"""
    if job_id not in workflow_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = workflow_jobs[job_id]
    return {
        "job_id": job_id,
        "session_id": job["session_id"],
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "started_at": job["started_at"],
        "completed_at": job.get("completed_at"),
        "error": job.get("error")
    }

# ============= SKILL ANALYSIS ENDPOINTS =============

@app.get("/api/interviews/{session_id}/skills", tags=["Skills"])
async def get_skill_gaps(session_id: str):
    """Get identified skill gaps"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_db[session_id]["session"]
    
    gaps = []
    for gap in session.skill_gaps:
        gaps.append({
            "skill_name": gap.skill_name,
            "required_level": gap.required_level,
            "importance": gap.importance.value,
            "gap_score": gap.gap_score,
            "description": gap.description,
            "examples": gap.examples
        })
    
    return {"skill_gaps": gaps}

# ============= QUESTION MANAGEMENT ENDPOINTS =============

@app.get("/api/interviews/{session_id}/questions", tags=["Questions"])
async def get_questions(session_id: str):
    """Get all questions for session"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_db[session_id]["session"]
    
    questions = []
    for q in session.questions_asked:
        questions.append({
            "id": q.id,
            "text": q.text,
            "type": q.question_type.value,
            "difficulty": q.difficulty,
            "skill_gap": q.skill_gap.skill_name if q.skill_gap else None
        })
    
    return {"questions": questions}

@app.get("/api/interviews/{session_id}/questions/{question_id}", tags=["Questions"])
async def get_question(session_id: str, question_id: str):
    """Get specific question"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_db[session_id]["session"]
    
    for q in session.questions_asked:
        if q.id == question_id:
            return {
                "id": q.id,
                "text": q.text,
                "type": q.question_type.value,
                "difficulty": q.difficulty,
                "skill_gap": q.skill_gap.skill_name if q.skill_gap else None,
                "follow_up_context": q.follow_up_context
            }
    
    raise HTTPException(status_code=404, detail="Question not found")

# ============= ANSWER SUBMISSION ENDPOINTS =============

@app.post("/api/interviews/{session_id}/answers", tags=["Answers"])
async def submit_answer(session_id: str, request: AnswerSubmitRequest):
    """Submit candidate answer"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_entry = sessions_db[session_id]
    session = db_entry["session"]
    
    # Find question
    question = None
    for q in session.questions_asked:
        if q.id == request.question_id:
            question = q
            break
    
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # In production, use Grader agent to score
    # For now, return mock scores
    
    return {
        "question_id": request.question_id,
        "status": "received",
        "message": "Answer received and will be graded"
    }

@app.get("/api/interviews/{session_id}/answers", tags=["Answers"])
async def get_answers(session_id: str):
    """Get all answers for session"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_db[session_id]["session"]
    
    answers = []
    for a in session.answers_provided:
        answers.append({
            "question_id": a.question_id,
            "clarity_score": a.clarity_score,
            "accuracy_score": a.technical_accuracy,
            "completeness_score": a.completeness_score,
            "overall_score": a.get_overall_score(),
            "needs_clarification": a.needs_clarification
        })
    
    return {"answers": answers}

# ============= ASSESSMENT RESULTS ENDPOINTS =============

@app.get("/api/interviews/{session_id}/results", response_model=AssessmentResultResponse, tags=["Results"])
async def get_assessment_results(session_id: str):
    """Get final assessment results"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_db[session_id]["session"]
    
    if not session.interview_complete:
        raise HTTPException(status_code=400, detail="Interview not yet completed")
    
    skill_gaps = []
    for gap in session.skill_gaps:
        skill_gaps.append({
            "skill_name": gap.skill_name,
            "required_level": gap.required_level,
            "importance": gap.importance.value,
            "gap_score": gap.gap_score,
            "description": gap.description
        })
    
    return AssessmentResultResponse(
        session_id=session.session_id,
        candidate_name=session.candidate_name,
        position_title=session.position_title,
        overall_score=session.overall_score,
        questions_count=len(session.questions_asked),
        answers_count=len(session.answers_provided),
        skill_gaps=skill_gaps,
        recommendations=session.recommendations,
        completed_at=datetime.now().isoformat()
    )

# ============= EXPORT ENDPOINTS =============

@app.get("/api/interviews/{session_id}/export/json", tags=["Export"])
async def export_as_json(session_id: str):
    """Export interview results as JSON"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_db[session_id]["session"]
    
    export_data = {
        "session_id": session.session_id,
        "candidate": session.candidate_name,
        "position": session.position_title,
        "overall_score": session.overall_score,
        "skill_gaps": [
            {
                "skill": g.skill_name,
                "importance": g.importance.value,
                "gap_score": g.gap_score
            }
            for g in session.skill_gaps
        ],
        "questions_count": len(session.questions_asked),
        "answers_count": len(session.answers_provided),
        "recommendations": session.recommendations,
        "created_at": session.created_at,
        "completed": session.interview_complete
    }
    
    return export_data

@app.get("/api/interviews/{session_id}/export/csv", tags=["Export"])
async def export_as_csv(session_id: str):
    """Export interview results as CSV"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_db[session_id]["session"]
    
    # Generate CSV
    csv_content = "Question,Type,Clarity,Accuracy,Completeness,Overall\n"
    
    for idx, (q, a) in enumerate(zip(session.questions_asked, session.answers_provided)):
        csv_content += f'"{q.text}",{q.question_type.value},{a.clarity_score:.2f},'
        csv_content += f'{a.technical_accuracy:.2f},{a.completeness_score:.2f},{a.get_overall_score():.2f}\n'
    
    return {
        "csv": csv_content,
        "filename": f"{session.candidate_name}_{session_id}.csv"
    }

# ============= AGENT STATISTICS ENDPOINTS =============

@app.get("/api/statistics", tags=["Statistics"])
async def get_statistics():
    """Get system statistics"""
    return {
        "total_sessions": len(sessions_db),
        "completed_sessions": sum(1 for s in sessions_db.values() if s["session"].interview_complete),
        "active_jobs": sum(1 for j in workflow_jobs.values() if j["status"] == "running"),
        "total_questions_generated": sum(len(s["session"].questions_asked) for s in sessions_db.values()),
        "total_answers_evaluated": sum(len(s["session"].answers_provided) for s in sessions_db.values()),
        "avg_score": (
            sum(s["session"].overall_score for s in sessions_db.values() if s["session"].interview_complete) /
            max(1, sum(1 for s in sessions_db.values() if s["session"].interview_complete))
        )
    }

# ============= DOCUMENTATION ENDPOINT =============

@app.get("/api/docs", tags=["Documentation"])
async def get_api_docs():
    """Get API documentation"""
    return {
        "title": "Multi-Agent Interview Assessment API",
        "version": "1.0.0",
        "endpoints": {
            "interviews": "/api/interviews",
            "workflows": "/api/workflows/execute",
            "skills": "/api/interviews/{session_id}/skills",
            "questions": "/api/interviews/{session_id}/questions",
            "answers": "/api/interviews/{session_id}/answers",
            "results": "/api/interviews/{session_id}/results"
        }
    }

# ============= ERROR HANDLERS =============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "status_code": exc.status_code,
        "detail": exc.detail,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
