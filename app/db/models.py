"""
ORM models — read-only mirror of the gateway's ``users`` table.
This service never writes to the users table; it only reads API keys.
"""
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    api_key = Column(String(255), unique=True, index=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    allowed_origins = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<User id={self.id} username='{self.username}'>"
