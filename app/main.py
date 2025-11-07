from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.db import Base, engine
from app.routes import auth_router, precheck_router, chat_router

# --- 모델 메타데이터 등록용 import ---
from app.models import user as user_model
# 다른 모델 코드와 병합 예정...
# from app.models import chat_thread as chat_thread_model
# from app.models import chat_message as chat_message_model
# from app.models import chat_attachment as chat_attachment_model
# from app.models import analysis_snapshot as analysis_snapshot_model
# from app.models import audit_log as audit_log_model

# --- 라우터 ---
from app.routes import auth, precheck, chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title="JeonSafe API",
    version="0.1.0",
    description="전세 계약 사기 위험도 분석, 증거 자료 관리, 법률 상담 지원을 위한 백엔드 REST API",
)

@app.get("/", summary="Health")
def health():
    return {"ok": True, "service": "JeonSafe API"}

# 라우터 등록
app.include_router(auth.router)
app.include_router(precheck.router)
app.include_router(chat.router)
