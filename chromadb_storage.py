"""
ChromaDB Integration for Vector Storage
Store and retrieve interview data, resumes, and job descriptions
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import json
from datetime import datetime
import uuid

# ============= CHROMADB CLIENT =============

class ChromaDBManager:
    """Manager for ChromaDB operations"""
    
    def __init__(self, persist_directory: str = "./chromadb_data"):
        """Initialize ChromaDB client"""
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False,
        )
        
        self.client = chromadb.Client(settings)
        self.persist_directory = persist_directory
        
        # Create collections
        self.resumes_collection = self.client.get_or_create_collection(
            name="resumes",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.job_descriptions_collection = self.client.get_or_create_collection(
            name="job_descriptions",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.interview_sessions_collection = self.client.get_or_create_collection(
            name="interview_sessions",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.questions_collection = self.client.get_or_create_collection(
            name="questions",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.answers_collection = self.client.get_or_create_collection(
            name="answers",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("✅ ChromaDB Initialized")
        print(f"   Collections: resumes, job_descriptions, interview_sessions, questions, answers")
        print(f"   Data directory: {persist_directory}")
    
    # ============= RESUME OPERATIONS =============
    
    def store_resume(self, session_id: str, candidate_name: str, resume_text: str) -> str:
        """Store resume in vector database"""
        doc_id = f"resume_{session_id}"
        
        self.resumes_collection.add(
            ids=[doc_id],
            documents=[resume_text],
            metadatas=[{
                "session_id": session_id,
                "candidate_name": candidate_name,
                "type": "resume",
                "timestamp": datetime.now().isoformat(),
                "text_length": len(resume_text)
            }]
        )
        
        print(f"✓ Stored resume for {candidate_name}")
        return doc_id
    
    def search_similar_resumes(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar resumes"""
        results = self.resumes_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_results(results)
    
    def get_resume(self, session_id: str) -> Optional[str]:
        """Retrieve specific resume"""
        results = self.resumes_collection.get(
            ids=[f"resume_{session_id}"]
        )
        
        if results["documents"]:
            return results["documents"][0]
        return None
    
    # ============= JOB DESCRIPTION OPERATIONS =============
    
    def store_job_description(self, session_id: str, position: str, jd_text: str) -> str:
        """Store job description in vector database"""
        doc_id = f"jd_{session_id}"
        
        self.job_descriptions_collection.add(
            ids=[doc_id],
            documents=[jd_text],
            metadatas=[{
                "session_id": session_id,
                "position": position,
                "type": "job_description",
                "timestamp": datetime.now().isoformat(),
                "text_length": len(jd_text)
            }]
        )
        
        print(f"✓ Stored job description for {position}")
        return doc_id
    
    def search_job_descriptions(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar job descriptions"""
        results = self.job_descriptions_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_results(results)
    
    # ============= INTERVIEW SESSION OPERATIONS =============
    
    def store_interview_session(self, session_id: str, session_data: Dict) -> str:
        """Store interview session metadata"""
        session_summary = f"""
        Candidate: {session_data.get('candidate_name')}
        Position: {session_data.get('position_title')}
        Session ID: {session_id}
        """
        
        self.interview_sessions_collection.add(
            ids=[f"session_{session_id}"],
            documents=[session_summary],
            metadatas=[{
                "session_id": session_id,
                "candidate_name": session_data.get('candidate_name'),
                "position": session_data.get('position_title'),
                "created_at": datetime.now().isoformat(),
                "status": session_data.get('status', 'active')
            }]
        )
        
        return session_id
    
    def search_interviews(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search interviews by candidate or position"""
        results = self.interview_sessions_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_results(results)
    
    def get_candidate_history(self, candidate_name: str) -> List[Dict]:
        """Get all interviews for a candidate"""
        results = self.interview_sessions_collection.get(
            where={"candidate_name": candidate_name}
        )
        
        if results["documents"]:
            return [
                {
                    "session_id": meta["session_id"],
                    "position": meta.get("position"),
                    "created_at": meta.get("created_at"),
                    "status": meta.get("status")
                }
                for meta in results["metadatas"]
            ]
        return []
    
    # ============= QUESTION OPERATIONS =============
    
    def store_question(self, session_id: str, question_data: Dict) -> str:
        """Store generated question"""
        doc_id = f"q_{session_id}_{question_data.get('id')}"
        
        self.questions_collection.add(
            ids=[doc_id],
            documents=[question_data.get('text', '')],
            metadatas=[{
                "session_id": session_id,
                "question_id": question_data.get('id'),
                "type": question_data.get('type', 'technical'),
                "difficulty": question_data.get('difficulty', 1),
                "skill_gap": question_data.get('skill_gap', ''),
                "generated_by": question_data.get('generated_by', 'interviewer'),
                "timestamp": datetime.now().isoformat()
            }]
        )
        
        return doc_id
    
    def get_questions_for_session(self, session_id: str) -> List[Dict]:
        """Get all questions for a session"""
        results = self.questions_collection.get(
            where={"session_id": session_id}
        )
        
        if results["documents"]:
            return [
                {
                    "id": meta["question_id"],
                    "text": doc,
                    "type": meta.get("type"),
                    "difficulty": meta.get("difficulty"),
                    "skill_gap": meta.get("skill_gap")
                }
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
        return []
    
    def search_questions_by_skill(self, skill: str, n_results: int = 10) -> List[Dict]:
        """Search questions for a specific skill"""
        results = self.questions_collection.get(
            where={"skill_gap": skill}
        )
        
        return [
            {
                "id": meta["question_id"],
                "text": doc,
                "difficulty": meta.get("difficulty")
            }
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
    
    # ============= ANSWER OPERATIONS =============
    
    def store_answer(self, session_id: str, answer_data: Dict) -> str:
        """Store candidate answer"""
        doc_id = f"a_{session_id}_{answer_data.get('question_id')}"
        
        self.answers_collection.add(
            ids=[doc_id],
            documents=[answer_data.get('text', '')],
            metadatas=[{
                "session_id": session_id,
                "question_id": answer_data.get('question_id'),
                "clarity_score": answer_data.get('clarity_score', 0.0),
                "accuracy_score": answer_data.get('accuracy_score', 0.0),
                "completeness_score": answer_data.get('completeness_score', 0.0),
                "overall_score": answer_data.get('overall_score', 0.0),
                "needs_clarification": str(answer_data.get('needs_clarification', False)),
                "timestamp": datetime.now().isoformat()
            }]
        )
        
        return doc_id
    
    def get_answers_for_session(self, session_id: str) -> List[Dict]:
        """Get all answers for a session"""
        results = self.answers_collection.get(
            where={"session_id": session_id}
        )
        
        if results["documents"]:
            return [
                {
                    "question_id": meta["question_id"],
                    "text": doc,
                    "clarity_score": meta.get("clarity_score", 0.0),
                    "accuracy_score": meta.get("accuracy_score", 0.0),
                    "completeness_score": meta.get("completeness_score", 0.0),
                    "overall_score": meta.get("overall_score", 0.0)
                }
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
        return []
    
    def find_high_quality_answers(self, min_score: float = 0.7) -> List[Dict]:
        """Find high-quality answers for training"""
        results = self.answers_collection.get()
        
        high_quality = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if float(meta.get("overall_score", 0)) >= min_score:
                high_quality.append({
                    "text": doc,
                    "score": meta.get("overall_score"),
                    "skill": meta.get("question_id")
                })
        
        return high_quality
    
    # ============= ANALYTICS & REPORTING =============
    
    def get_collection_stats(self) -> Dict:
        """Get statistics for all collections"""
        return {
            "resumes": {
                "count": self.resumes_collection.count(),
                "name": "resumes"
            },
            "job_descriptions": {
                "count": self.job_descriptions_collection.count(),
                "name": "job_descriptions"
            },
            "interview_sessions": {
                "count": self.interview_sessions_collection.count(),
                "name": "interview_sessions"
            },
            "questions": {
                "count": self.questions_collection.count(),
                "name": "questions"
            },
            "answers": {
                "count": self.answers_collection.count(),
                "name": "answers"
            }
        }
    
    def export_session_to_json(self, session_id: str) -> Dict:
        """Export complete session data"""
        resume = self.get_resume(session_id)
        questions = self.get_questions_for_session(session_id)
        answers = self.get_answers_for_session(session_id)
        
        return {
            "session_id": session_id,
            "resume": resume,
            "questions": questions,
            "answers": answers,
            "exported_at": datetime.now().isoformat()
        }
    
    # ============= UTILITY METHODS =============
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results"""
        formatted = []
        
        if results["documents"] and results["metadatas"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": meta
                })
        
        return formatted
    
    def clear_database(self):
        """Clear all data (use with caution)"""
        self.client.delete_collection(name="resumes")
        self.client.delete_collection(name="job_descriptions")
        self.client.delete_collection(name="interview_sessions")
        self.client.delete_collection(name="questions")
        self.client.delete_collection(name="answers")
        
        print("⚠️ All data cleared from ChromaDB")
    
    def backup_to_json(self, filepath: str = "chromadb_backup.json"):
        """Backup all data to JSON"""
        backup = {
            "timestamp": datetime.now().isoformat(),
            "collections": {
                "resumes": self._get_collection_data("resumes"),
                "job_descriptions": self._get_collection_data("job_descriptions"),
                "interview_sessions": self._get_collection_data("interview_sessions"),
                "questions": self._get_collection_data("questions"),
                "answers": self._get_collection_data("answers")
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(backup, f, indent=2)
        
        print(f"✓ Backup saved to {filepath}")
        return filepath
    
    def _get_collection_data(self, collection_name: str) -> List[Dict]:
        """Get all data from a collection"""
        collection = self.client.get_collection(name=collection_name)
        results = collection.get()
        
        data = []
        if results["documents"]:
            for doc, meta in zip(results["documents"], results["metadatas"]):
                data.append({
                    "text": doc,
                    "metadata": meta
                })
        
        return data

# ============= SINGLETON INSTANCE =============

_db_instance: Optional[ChromaDBManager] = None

def get_chromadb_manager() -> ChromaDBManager:
    """Get ChromaDB manager instance (singleton)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = ChromaDBManager()
    return _db_instance

# ============= TEST DEMO =============

def demo_chromadb():
    """Demonstrate ChromaDB operations"""
    print("\n" + "="*70)
    print("📦 ChromaDB Demo")
    print("="*70)
    
    db = ChromaDBManager()
    
    # Store sample data
    session_id = str(uuid.uuid4())
    
    db.store_resume(session_id, "John Doe", "Python, REST API, Docker, Kubernetes")
    db.store_job_description(session_id, "Backend Engineer", "5+ years Python, REST APIs, Kubernetes")
    
    # Store questions
    db.store_question(session_id, {
        "id": "q1",
        "text": "Explain your experience with Python",
        "type": "technical",
        "difficulty": 2,
        "skill_gap": "python"
    })
    
    # Store answer
    db.store_answer(session_id, {
        "question_id": "q1",
        "text": "I have 8 years of Python experience...",
        "clarity_score": 0.85,
        "accuracy_score": 0.9,
        "completeness_score": 0.88,
        "overall_score": 0.87
    })
    
    # Display stats
    stats = db.get_collection_stats()
    print("\n📊 Collection Statistics:")
    for name, data in stats.items():
        print(f"   • {data['name']}: {data['count']} items")
    
    print("\n✅ Demo Complete!")

if __name__ == "__main__":
    demo_chromadb()
