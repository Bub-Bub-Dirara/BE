# 빈 파일에서 수정하였습니다 11/5 feature/db
from .auth import router as auth_router
from .precheck import router as precheck_router
from .chat import router as chat_router

__all__ = ["auth_router", "precheck_router", "chat_router"]