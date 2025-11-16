from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from be.app.core.db import Base, engine
from be.app.core.config import settings
from be.app.routes import auth, precheck, chat, upload
from be.app.models import user as user_model  # 모델 로딩용 (사용은 안 해도 됨)

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 라이프사이클 관리:
    - startup 시: 테이블 생성 + S3 설정 로그
    - shutdown 시: 현재는 별도 처리 없음
    """
    # startup
    Base.metadata.create_all(bind=engine)
    logger.info(
        f"S3 enabled={settings.s3_enabled} "
        f"region={settings.aws_region} bucket={settings.s3_bucket}"
    )
    yield
    # shutdown (필요하면 여기서 정리 작업)


app = FastAPI(
    title="JeonSafe API",
    version="0.1.0",
    description="전세 계약 사기 위험도 분석, 증거 자료 관리, 법률 상담 지원을 위한 백엔드 REST API",
    lifespan=lifespan,  # ← lifespan 등록
)

# CORS 설정
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # 프론트엔드 도메인 허용
    allow_credentials=True,    # 쿠키/인증 헤더 사용
    allow_methods=["*"],       # 모든 메서드 허용 (GET, POST 등)
    allow_headers=["*"],       # 모든 헤더 허용
)


@app.get("/", summary="Health")
def health():
    return {"ok": True, "service": "JeonSafe API"}


# 라우터 등록
app.include_router(auth.router)
app.include_router(precheck.router)
app.include_router(chat.router)
app.include_router(upload.router)
