from ..ai import AIModel, AIModelMessage
from pydantic import Field
from .api import OpenAIClient, OpenAIMessage, ChatCompletionRequest, UsageContext, Usage
from typing import AsyncIterator, Optional

def _messages_to_openai_messages(message: list[AIModelMessage]) -> list[OpenAIMessage]:
    def _message_to_openai_message(message: AIModelMessage) -> OpenAIMessage:
        return OpenAIMessage(content=message.content, role=message.role, name=message.name)
    return [_message_to_openai_message(message) for message in message]

class OpenAIModel(AIModel):
    model: str
    temperature: float = 0.3
    openai_client: OpenAIClient
    usage_context: UsageContext = Field(default_factory=UsageContext)

    @property
    def usage(self) -> Usage:
        return self.usage_context.usage[self.model]

    async def generate_multiple(self, history: list[AIModelMessage], stop: list[str], n: int, max_tokens: Optional[int] = None) -> list[str]:
        openai_messages = _messages_to_openai_messages(history)
        req = ChatCompletionRequest(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            stop=stop,
            n=n,
            max_tokens=max_tokens
        )
        response = await self.openai_client.chat_completion(req, self.usage_context)
        return [choice.message.content for choice in response.choices]

    async def generate_stream(self, history: list[AIModelMessage], stop: list[str], max_tokens: Optional[int] = None) -> AsyncIterator[str]:
        openai_messages = _messages_to_openai_messages(history)
        req = ChatCompletionRequest(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            stop=stop,
            n=1,
            max_tokens=max_tokens
        )
        async for chunk in self.openai_client.streaming_chat_completion(req, self.usage_context):
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content

    def override(self, *, temperature: float) -> AIModel:
        return OpenAIModel(**self.dict(), temperature=temperature)
