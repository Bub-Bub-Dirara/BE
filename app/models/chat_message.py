from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.sql import func
from app.core.db import Base

class ChatMessage(Base):
    __tablename__ = "chat_message"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey("chat_thread.id", ondelete="CASCADE"))
    role = Column(String, nullable=False)      # user / assistant / system
    content = Column(String, nullable=False)
    tokens_in = Column(Integer)
    tokens_out = Column(Integer)
    step = Column(String)     # 예: UPLOAD / MATCH / REPORT
    meta = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
