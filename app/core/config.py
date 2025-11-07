import secrets
from datetime import timedelta
import os

# 실제 운영에선 .env로 분리 권장
SECRET_KEY = secrets.token_urlsafe(32)  # 개발용 자동 생성
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 60분
ACCESS_TOKEN_EXPIRE_DELTA = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

class Settings:
    APP_ENV = os.getenv("APP_ENV", "local")

    AWS_REGION = os.getenv("AWS_REGION")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE")  # 퍼블릭 URL 프리픽스(정책에 따라 미사용 가능)

    # 20MB 기본
    MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))

    @property
    def s3_enabled(self) -> bool:
        return all([self.AWS_REGION, self.AWS_ACCESS_KEY_ID, self.AWS_SECRET_ACCESS_KEY, self.S3_BUCKET])

settings = Settings()