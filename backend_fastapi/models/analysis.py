from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from backend_fastapi.core.database import Base
import uuid

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=True)
    org_id = Column(String, nullable=True)
    filename = Column(String(255))
    config = Column(JSON)
    status = Column(String(50), default="pending")
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id"))
    best_model = Column(String(100))
    best_score = Column(Float)
    metrics = Column(JSON)
    feature_importance = Column(JSON)
    shap_summary = Column(JSON)
    predictions_sample = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
