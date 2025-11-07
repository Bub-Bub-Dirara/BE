import os
import uuid
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from .config import settings

class S3Client:
    def __init__(self):
        if not settings.s3_enabled:
            self.client = None
            return
        self.client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    def upload_bytes(self, data: bytes, key: str, content_type: str) -> str:
        if not self.client:
            raise RuntimeError("S3 not configured")
        try:
            self.client.put_object(
                Bucket=settings.S3_BUCKET,
                Key=key,
                Body=data,
                ContentType=content_type,
                ACL="private",  # 필요 시 'public-read'
            )
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"S3 upload failed: {e}")
        # 퍼블릭 URL 정책을 쓰지 않는다면 presigned URL을 쓰도록
        return f"{settings.S3_PUBLIC_BASE}/{key}" if settings.S3_PUBLIC_BASE else key

    def generate_presigned_url(self, key: str, expires: int = 3600) -> str:
        if not self.client:
            raise RuntimeError("S3 not configured")
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET, "Key": key},
                ExpiresIn=expires,
            )
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"S3 presign failed: {e}")

s3 = S3Client()

def build_s3_key(user_id: int, category: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return f"users/{user_id}/{category}/{uuid.uuid4().hex}{ext}"
