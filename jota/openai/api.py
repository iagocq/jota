from pydantic import BaseModel, Field, validate_arguments
from typing import Optional, Literal, Any, AsyncIterator
import httpx
import re
import logging
import tiktoken
from io import StringIO
from collections import defaultdict
logger = logging.getLogger(__name__)

class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: Optional[str]

class OpenAIFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Any

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    functions: Optional[list[OpenAIFunction]] = None
    function_call: Optional[Literal['none', 'auto']] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[list[str] | str] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: Literal['stop', 'length', 'function_call']

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __iadd__(self, other: 'Usage') -> 'Usage':
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        return self

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

class ChoiceChunk(BaseModel):
    class Delta(BaseModel):
        role: Optional[str]
        content: Optional[str]

    index: int
    delta: Delta
    finish_reason: Optional[Literal['stop', 'length', 'function_call']]

class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChoiceChunk]

class Pricing(BaseModel):
    input: float
    output: float
    unit: Literal['1K tokens'] = '1K tokens'
    pricing_unit: Literal['USD'] = 'USD'

GPT_3_5_TURBO_4K_PRICING = Pricing(input=0.0015, output=0.002)
GPT_3_5_TURBO_16K_PRICING = Pricing(input=0.003, output=0.004)
GPT_4_8K_PRICING = Pricing(input=0.03, output=0.06)
GPT_4_32K_PRICING = Pricing(input=0.06, output=0.12)

DEFAULT_PRICING: dict[str, Pricing] = {
    'gpt-3.5-turbo-16k': GPT_3_5_TURBO_16K_PRICING,
    'gpt-3.5-turbo': GPT_3_5_TURBO_4K_PRICING,
    'gpt-4': GPT_4_8K_PRICING,
}

class ModelUsage(BaseModel):
    model: str
    usage: Usage

class UsageContext(BaseModel):
    usages: list[ModelUsage] = Field(default_factory=list)
    pricing: dict[str, Pricing] = DEFAULT_PRICING

    @property
    def usage(self) -> dict[str, Usage]:
        usages: dict[str, Usage] = defaultdict(Usage)
        for model_usage in self.usages:
            model = model_usage.model
            usage = model_usage.usage
            usages[model] += usage
        return usages

class OpenAIError(Exception): ...

class OpenAIRequestError(OpenAIError):
    message: str
    type: str
    code: str
    param: Optional[str]

    @validate_arguments
    def __init__(self, message: str, type: str, code: str, param: Optional[str]) -> None:
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        super().__init__(self.__str__())

    def __str__(self) -> str:
        return f'{self.type} {self.code}: {self.message}'

class OpenAIClient(BaseModel):
    api_key: str
    api_url: str = "https://api.openai.com/v1"
    http_client: httpx.AsyncClient = Field(default_factory=lambda: httpx.AsyncClient(timeout=120))

    class Config:
        arbitrary_types_allowed = True

    def _openai_check_error(self, json: dict[str, Any]) -> None:
        if 'error' in json:
            err = json['error']
            raise OpenAIRequestError(message=err['message'],
                                     type=err['type'],
                                     code=err['code'],
                                     param=err.get('param'))

    async def _openai_request(self, method: str, endpoint: str, data: Optional[dict] = None) -> Any:
        logger.debug(f'OpenAI request: {method} {endpoint} {data}')
        response = await self.http_client.request(
            method,
            f'{self.api_url}{endpoint}',
            headers=self.headers,
            json=data,
        )
        json = response.json()
        self._openai_check_error(json)
        return json

    async def _openai_stream(self, method: str, endpoint: str, data: Optional[dict] = None) -> AsyncIterator[str]:
        logger.debug(f'OpenAI stream: {method} {endpoint} {data}')
        async with self.http_client.stream(
            method,
            f'{self.api_url}{endpoint}',
            headers=self.headers,
            json=data,
        ) as stream:
            if stream.status_code != 200:
                await stream.aread()
                self._openai_check_error(stream.json())
            async for chunk in stream.aiter_lines():
                logger.debug(f'OpenAI stream chunk: {chunk}')
                yield chunk

    async def chat_completion(self,
                              req: ChatCompletionRequest,
                              usage_context: Optional[UsageContext] = None) -> ChatCompletionResponse:
        logger.info(f'Chat completion request: {req}')
        request = req.dict(exclude_none=True)
        raw_response = await self._openai_request('POST', '/chat/completions', request)
        response = ChatCompletionResponse.parse_obj(raw_response)

        if usage_context is not None:
            logger.debug(f'Updating usage context with {response.usage}')
            usage_context.usages.append(ModelUsage(model=req.model, usage=response.usage))

        return response

    class TokenCount(BaseModel):
        per_message: int
        per_name: int

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    TOKEN_COUNTS: dict[str, TokenCount] = {
        'gpt-3.5-turbo-0301': TokenCount(per_message=4, per_name=-1),
        'gpt-3.5-turbo': TokenCount(per_message=3, per_name=1),
        'gpt-4': TokenCount(per_message=3, per_name=1),
    }

    _chunk_re = re.compile(r'data: (.*)')
    async def streaming_chat_completion(self,
                                        req: ChatCompletionRequest,
                                        usage_context: Optional[UsageContext] = None) -> AsyncIterator[ChatCompletionChunk]:
        logger.info(f'Streaming chat completion request: {req}')

        request = req.dict(exclude_none=True)
        request['stream'] = True

        chunks: list[ChatCompletionChunk] = []

        async for chunk in self._openai_stream('POST', '/chat/completions', request):
            m = self._chunk_re.match(chunk)
            if not m: continue
            data = m.group(1)
            if data == '': continue
            if data == '[DONE]': break
            chunk = ChatCompletionChunk.parse_raw(data)
            chunks.append(chunk)
            yield chunk

        tokenizer = tiktoken.encoding_for_model(req.model)
        completion_tokens = 0
        choices: dict[int, StringIO] = defaultdict(StringIO)

        for chunk in chunks:
            for choice in chunk.choices:
                content = choice.delta.content
                if content is None: continue
                choices[choice.index].write(content)

        for choice in choices.values():
            full_str = choice.getvalue()
            tokens = tokenizer.encode(full_str)
            completion_tokens += len(tokens)

        for k, v in self.TOKEN_COUNTS.items():
            if k in req.model:
                model_token_count = v
                break
        else:
            model_token_count = self.TOKEN_COUNTS['gpt-4']

        prompt_tokens = 0
        for message in req.messages:
            prompt_tokens += model_token_count.per_message
            prompt_tokens += len(tokenizer.encode(message.content))
            prompt_tokens += len(tokenizer.encode(message.role))
            if message.name:
                prompt_tokens += len(tokenizer.encode(message.name))
                prompt_tokens += model_token_count.per_name

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens)
        
        if usage_context is not None:
            logger.debug(f'Updating usage context with {usage}')
            usage_context.usages.append(ModelUsage(model=req.model, usage=usage))

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}
