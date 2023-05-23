"""Schemas for the chat app."""
from typing import List, Optional

from pydantic import BaseModel


class ChatResponse(BaseModel):
    """Chat response schema."""

    token: Optional[str] = None
    error: Optional[str] = None
    sourceDocs: Optional[List] = None
