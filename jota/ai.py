from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Any

class AIModelMessage(BaseModel):
    role: str
    name: Optional[str] = None
    content: str

class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Any

class AIModel(BaseModel, ABC):
    async def generate(self, history: list[AIModelMessage], *, stop: list[str] = [], max_tokens: Optional[int] = None) -> str:
        generations = await self.generate_multiple(history, stop=stop, n=1, max_tokens=max_tokens)
        return generations[0]

    @abstractmethod
    async def generate_multiple(self, history: list[AIModelMessage], *, stop: list[str] = [], n: int, max_tokens: Optional[int] = None) -> list[str]: ...

    @abstractmethod
    async def generate_stream(self, history: list[AIModelMessage], *, stop: list[str] = [], max_tokens: Optional[int] = None) -> AsyncIterator[str]:
        yield ''
