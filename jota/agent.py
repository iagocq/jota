from pydantic import BaseModel, Field
from typing import AsyncIterator, Optional, Any
from .ai import History, AIModel, AIMessage, ChatMessage, HintMessage

class Template(BaseModel):
    template: str

    def format(self, *args: Any, **kwargs: Any) -> str:
        return self.template.format(*args, **kwargs)

CHAT_TYPE = {
    'group': "You're a member of a group chat.",
    '1-to-1': "You're chatting with someone.",
}

class ConversationalAgent(BaseModel):
    prompt: Template = Template(template=
        "{chat_type}"
        " Your name is {name}."
        " You're relaxed, friendly and helpful. "
        " You have access to the last few messages in the chat. "
        " Your main language is {language}."
        " Use search results to enhance your anwers."
        " Only provide answers about courses, professors, and other academic information based on the search results."
        "\n\nMiscellaneous information:\n{information}"
    )
    history: History = Field(default_factory=lambda: History(messages=[]))
    model: AIModel
    is_group_chat: bool = False
    name: str = 'Jota'
    language: str = 'portuguese'
    information: list[str] = Field(default_factory=list)
    max_history_messages: int = 20
    max_history_characters: int = 2000

    def _prepare_history(self, user_msg: ChatMessage, hint_msg: Optional[HintMessage]) -> tuple[list[AIMessage], ChatMessage]:
        ai_msgs: list[AIMessage] = []

        prompt = self.prompt.format(
            chat_type=CHAT_TYPE['group' if self.is_group_chat else '1-to-1'],
            name=self.name,
            language='portuguese',
            information='\n'.join(f'- {info}' for info in self.information)
        )
        prompt_msg = AIMessage(role='system', content=prompt)

        ai_msgs.append(prompt_msg)

        history_msgs = (
            self.history
                .last_n_messages(self.max_history_messages)
                .limit_characters(self.max_history_characters)
                .to_messages())

        ai_msgs.extend(history_msgs)

        self.history.add_message(user_msg)
        ai_msgs.append(user_msg.to_message())
        if hint_msg:
            ai_msgs.append(hint_msg.to_message())
            self.history.add_message(hint_msg)

        ai_chat_msg = ChatMessage(
            sender='assistant',
            content='',
            in_reply_to=user_msg.id
        )
        self.history.add_message(ai_chat_msg)

        ai_msg = AIMessage(
            role='assistant',
            content=str(ai_chat_msg)
        )
        ai_msgs.append(ai_msg)

        return ai_msgs, ai_chat_msg

    async def reply_to_message(self, message: ChatMessage, hint: Optional[HintMessage] = None) -> str:
        messages, ai_chat_msg = self._prepare_history(message, hint)
        reply = await self.model.generate(messages, ['"""'])
        ai_chat_msg.content = reply

        return ai_chat_msg.content

    async def streaming_reply_to_message(self, message: ChatMessage, hint: Optional[HintMessage] = None) -> AsyncIterator[str]:
        messages, ai_chat_msg = self._prepare_history(message, hint)

        full_response: list[str] = []
        async for response in self.model.generate_stream(messages, ['"""']):
            full_response.append(response)
            yield response

        ai_chat_msg.content = ''.join(full_response)

class ClassificationAgent(BaseModel):
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
        prompt_msg = AIMessage(role='system', content=prompt)
        user_msg = AIMessage(role='user', content=msg)
        ai_msg = AIMessage(role='assistant', content='Category:')
        response = await self.model.generate([prompt_msg, user_msg, ai_msg], [], max_tokens=2)
        for category in self.categories.keys():
            if category in response:
                return category
        return None

class QueryGeneratorAgent(BaseModel):
    prompt = Template(template=
        "Generate a single {engine} query to the database that answers the user's prompt.\n"
        "Use the history messages to provide context to the query.\n\n"
        "If more than one {engine} query would be required, refuse to answer.\n"
        "If there is not enough information to generate a valid query, refuse to answer.\n"
        "Your response should be in the following format:\n\n"
        "StepByStep: [extract relevant table names, relevant given information, relevant columns. Give a step by step reasoning of the parts that make up the query.]\n\n"
        "SQLQuery: [a single {engine} query that answers the prompt. Only use tables and columns described in the database schema. When comparing names, always use LIKE and %, avoid using = in this case]\n\n"
    )
    model: AIModel
    engine: str
    db_schema: str
    history: History

    async def generate(self, user_prompt: str) -> str:
        prompt_msg = AIMessage(role='system', content=self.prompt.format(engine=self.engine))
        schema_msg = AIMessage(role='system', name="database_schema", content=self.db_schema)
        user_msg = AIMessage(role='user', content=user_prompt)
        response = await self.model.generate([prompt_msg, schema_msg, user_msg], [])
        return response
