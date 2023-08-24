from pydantic import BaseModel, Field
from datetime import datetime
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Any, Self

class AIMessage(BaseModel):
    role: str
    name: Optional[str] = None
    content: str

class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Any

class ChatMessage(BaseModel):
    id: Optional[int] = None
    sender: Optional[str]
    sent_at: datetime = Field(default_factory=datetime.now)
    content: str
    in_reply_to: Optional[int] = None

    def __str__(self) -> str:
        in_reply_to = f' (in reply to message {self.in_reply_to})' if self.in_reply_to is not None else ''
        sender = f'"{self.sender}"' if self.sender != 'assistant' else 'assistant'
        date = self.sent_at.strftime('%Y-%m-%d %H:%M:%S')
        content = f' """{self.content}"""' if self.content else ' """'
        return f'message {self.id} sent at {date} by {sender}{in_reply_to}:{content}'

    def to_message(self) -> AIMessage:
        role = 'assistant' if self.sender == 'assistant' else 'user'
        return AIMessage(role=role, name='chat_history_message', content=str(self))

class HintMessage(ChatMessage):
    def __str__(self) -> str:
        return self.content

    def to_message(self) -> AIMessage:
        return AIMessage(role='system', name='search_result', content=self.content)

class AIModel(BaseModel, ABC):
    async def generate(self, history: list[AIMessage], stop: list[str], max_tokens: Optional[int] = None) -> str:
        generations = await self.generate_multiple(history, stop, 1, max_tokens)
        return generations[0]

    @abstractmethod
    async def generate_multiple(self, history: list[AIMessage], stop: list[str], n: int, max_tokens: Optional[int] = None) -> list[str]: ...

    @abstractmethod
    async def generate_stream(self, history: list[AIMessage], stop: list[str]) -> AsyncIterator[str]:
        yield ''

class History(BaseModel):
    messages: list[ChatMessage]

    def get_message(self, id: int) -> Optional[ChatMessage]:
        for message in self.messages:
            if message.id == id:
                return message
        return None

    def last_n_messages(self, n: int, *, follow_replies_first: bool = True) -> Self:
        last = self.messages[-n:]

        if not follow_replies_first:
            return History(messages=last)

        messages = []
        messages_stack = last
        included = set()
        while len(messages_stack) > 0:
            message = messages_stack.pop()
            if message.id in included: continue
            included.add(message.id)
            messages.append(message)
            if message.in_reply_to is not None:
                reply = self.get_message(message.in_reply_to)
                if reply is not None:
                    messages_stack.append(reply)
            if len(messages) >= n:
                break

        messages.sort(key=lambda message: message.id)
        return History(messages=messages)

    def limit_characters(self, limit: int) -> Self:
        characters = 0
        messages = []
        for message in reversed(self.messages):
            characters += len(message.content)
            if characters > limit:
                break
            messages.append(message)

        messages.reverse()
        return History(messages=messages)

    def __str__(self) -> str:
        return '\n\n'.join([str(message) for message in self.messages])

    def to_messages(self) -> list[AIMessage]:
        return [msg.to_message() for msg in self.messages]

    def add_message(self, message: ChatMessage):
        if message.id is None:
            message.id = len(self.messages)
        self.messages.append(message)
