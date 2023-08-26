from pydantic import BaseModel, Field
from typing import AsyncIterator, Optional, Any, Callable, Awaitable, Self
from .ai import AIModel, AIModelMessage
from datetime import datetime

class Template(BaseModel):
    template: str

    def format(self, *args: Any, **kwargs: Any) -> str:
        return self.template.format(*args, **kwargs)

CHAT_TYPE = {
    'group': "You're a member of a group chat.",
    '1-to-1': "You're chatting with someone.",
}

class Message(BaseModel):
    content: str
    role: str
    name: str
    id: Optional[int] = None

    def to_message(self) -> AIModelMessage:
        return AIModelMessage(role=self.role, name=self.name, content=self.__str__())

class ChatMessage(Message):
    sender: Optional[str] = None
    sent_at: datetime = Field(default_factory=datetime.now)
    in_reply_to: Optional[int] = None
    name: str = 'chat_history_message'

    def __str__(self) -> str:
        nl = '\n'
        return f'''
message number: {self.id}
sent at: {self.sent_at:%Y-%m-%d %H:%M:%S}
author: {self.sender or self.role}
{f'in reply to message {self.in_reply_to}{nl}' if self.in_reply_to else ''}
###

{self.content}'''.strip()

class HintMessage(Message):
    role = 'system'

    def __str__(self) -> str:
        return self.content

class History(BaseModel):
    messages: list[Message]

    def get_message(self, id: int) -> Optional[Message]:
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
            if isinstance(message, ChatMessage) and message.in_reply_to is not None:
                reply = self.get_message(message.in_reply_to)
                if reply is not None:
                    messages_stack.append(reply)

            messages.append(message)

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
        return '\n------\n'.join([str(message) for message in self.messages])

    def to_messages(self) -> list[AIModelMessage]:
        return [msg.to_message() for msg in self.messages]

    def add_message(self, message: Message):
        if message.id is None:
            message.id = len(self.messages)
        self.messages.append(message)

class HistoryView(BaseModel):
    base_history: History = Field(default_factory=lambda: History(messages=[]))
    max_history_messages: int = 20
    max_history_characters: int = 2000

    @property
    def history(self) -> History:
        return (self.base_history
            .last_n_messages(self.max_history_messages)
            .limit_characters(self.max_history_characters)
        )

    @property
    def messages(self) -> list[AIModelMessage]:
        return self.history.to_messages()

    def add_message(self, message: Message):
        self.base_history.add_message(message)

    def __str__(self) -> str:
        return self.history.__str__()

class Classifier(BaseModel):
    prompt = Template(template=
        "Classify messages into one of the following categories:\n"
        "{categories}"
    )
    categories: dict[str, str]
    model: AIModel

    async def classify(self, msg: str) -> Optional[str]:
        prompt = self.prompt.format(
            categories=''.join(f'- {category}: {description}\n' for category, description in self.categories.items())
        )
        prompt_msg = AIModelMessage(role='system', content=prompt)
        user_msg = AIModelMessage(role='user', content=msg)
        ai_msg = AIModelMessage(role='assistant', content='Category:')
        response = await self.model.generate([prompt_msg, user_msg, ai_msg], max_tokens=2)
        for category in self.categories.keys():
            if category in response:
                return category
        return None

HinterFn = Callable[[Message, HistoryView], Awaitable[HintMessage]]

class Hinter(BaseModel):
    generators: dict[str, HinterFn]

    async def generate_hint(self, category: str, message: Message, context: HistoryView) -> Optional[HintMessage]:
        if category not in self.generators: return None
        hinter = self.generators[category]
        return await hinter(message, context)

class ConversationalAgent(BaseModel):
    prompt: Template = Template(template=
        "{chat_type}"
        " Your name is {name}."
        " You're relaxed, friendly and funny. "
        " You have access to the last few messages in the chat. "
        " Your main language is {language}."
        " Use search results to enhance your anwers."
        " Only provide answers about courses, professors, and other academic information based on the search results."
        " You have access to the current time."
        "\n\nMiscellaneous information:\n{information}"
    )
    history_view: HistoryView
    model: AIModel
    is_group_chat: bool = False
    name: str = 'Jota'
    language: str = 'portuguese'
    information: list[str] = Field(default_factory=list)
    classifier: Optional[Classifier] = None
    hinter: Optional[Hinter] = None

    def _prepare_history(self, user_msg: ChatMessage, hint_msg: Optional[HintMessage]) -> tuple[list[AIModelMessage], ChatMessage]:
        ai_msgs: list[AIModelMessage] = []

        prompt = self.prompt.format(
            chat_type=CHAT_TYPE['group' if self.is_group_chat else '1-to-1'],
            name=self.name,
            language=self.language,
            information='\n'.join(f'- {info}' for info in self.information)
        )
        prompt_msg = AIModelMessage(role='system', content=prompt)

        ai_msgs.append(prompt_msg)

        history_msgs = self.history_view.messages
        ai_msgs.extend(history_msgs)

        self.history_view.add_message(user_msg)
        ai_msgs.append(user_msg.to_message())
        if hint_msg:
            ai_msgs.append(hint_msg.to_message())
            self.history_view.add_message(hint_msg)

        ai_chat_msg = ChatMessage(
            role='assistant',
            content='',
            in_reply_to=user_msg.id
        )
        self.history_view.add_message(ai_chat_msg)

        ai_msg = ai_chat_msg.to_message()
        ai_msgs.append(ai_msg)

        return ai_msgs, ai_chat_msg

    async def _get_hint_for_message(self, message: ChatMessage) -> Optional[HintMessage]:
        if not self.classifier: return None
        if not self.hinter: return None

        category = await self.classifier.classify(message.content)
        if not category: return None

        return await self.hinter.generate_hint(category, message, self.history_view)

    async def reply_to_message(self, message: ChatMessage) -> str:
        hint = await self._get_hint_for_message(message)
        messages, ai_chat_msg = self._prepare_history(message, hint)
        reply = await self.model.generate(messages)
        ai_chat_msg.content = reply

        return ai_chat_msg.content

    async def streaming_reply_to_message(self, message: ChatMessage) -> AsyncIterator[str]:
        hint = await self._get_hint_for_message(message)
        messages, ai_chat_msg = self._prepare_history(message, hint)

        full_response: list[str] = []
        async for response in self.model.generate_stream(messages):
            full_response.append(response)
            yield response

        ai_chat_msg.content = ''.join(full_response)
