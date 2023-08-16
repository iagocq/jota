from langchain.chains.base import Chain
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import SystemMessage, AIMessage, HumanMessage, BaseMessage
from typing import Any, Optional
from . import prompt_gpt4 as prompt
from .db import SQLDatabase
import re
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

SYSTEM_COLOR = "blue"
USER_COLOR = "yellow"
AI_COLOR = "green"
SQL_COLOR = "red"

class AIAttempt(BaseModel):
    sql_query: Optional[str] = None
    answer: Optional[str] = None
    action: Optional[str] = None
    step_by_step: Optional[str] = None
    full_content: str
    human_message: HumanMessage

class SQLResult(BaseModel):
    sql_result: str
    sql_error: bool

class FailedAttempt(BaseModel):
    attempt: AIAttempt
    result: SQLResult

def parse_action(action: Optional[str]) -> str:
    if action is None or 'information' in action.lower():
        return 'user'
    elif 'retry' in action.lower():
        return 'retry'
    elif 'success' in action.lower():
        return 'success'
    else:
        return 'user'

_steps_re = re.compile(r'(\w+):\s+(.*?)(?=\n\w+:|$)', re.DOTALL)
_query_re = re.compile(r'^(```(sql(ite)?)?)?(?P<query>.*?)(```)?$', re.DOTALL | re.IGNORECASE)
def separate_steps(message: str) -> dict[str, str]:
    parts = {}
    steps = _steps_re.findall(message)
    for step, answer in steps:
        parts[step] = answer.strip()
    return parts

class SQLChain(Chain):
    llm: BaseLanguageModel
    db: SQLDatabase
    database_description: str
    output_key: str = "response"

    @property
    def input_keys(self) -> list[str]:
      return ["prompt"]

    @property
    def output_keys(self) -> list[str]:
      return [self.output_key]
  
    def print_msg(self, msg: BaseMessage | str, run_manager: Optional[CallbackManagerForChainRun] = None):
        if isinstance(msg, str):
            color = SQL_COLOR
            if not msg.endswith('\n'):
                content = f'{msg}\n'
            else:
                content = msg
        elif isinstance(msg, BaseMessage):
            content = msg.content
            if not content.endswith('\n'):
                content = f'{content}\n'
            if isinstance(msg, SystemMessage):
                color = SYSTEM_COLOR
            elif isinstance(msg, HumanMessage):
                color = USER_COLOR
            elif isinstance(msg, AIMessage):
                color = AI_COLOR
            else:
                raise TypeError(f"Unknown message type {msg.__class__.__name__}")

        if isinstance(msg, SystemMessage): return
        if run_manager is not None:
            run_manager.on_text(content, color=color, verbose=self.verbose)

    def print_msgs(self, msg: list[BaseMessage | str], run_manager: Optional[CallbackManagerForChainRun] = None):
        for m in msg:
            self.print_msg(m, run_manager)

    def _generate_query(self, user_prompt: str, previous_attempts: list[FailedAttempt], run_manager: Optional[CallbackManagerForChainRun] = None) -> AIAttempt:
        p = prompt.GEN_QUERY_PROMPT.format(
            database_description=self.database_description
        )
        gen_query_prompt = SystemMessage(content=p)

        u_prompt = HumanMessage(content=f'{user_prompt.strip()}\n')
        self.print_msgs([gen_query_prompt, u_prompt], run_manager)

        ai_response = self.llm.predict_messages(
            messages=[gen_query_prompt, u_prompt],
            stop=["\nSQLResult:"]
        )

        self.print_msg(ai_response, run_manager)

        steps = separate_steps(ai_response.content)
        sql_query = steps.get('SQLQuery')
        step_by_step = steps.get('StepByStep')
        answer = steps.get('Answer')
        return AIAttempt(sql_query=sql_query, answer=answer, step_by_step=step_by_step, full_content=ai_response.content, human_message=u_prompt)

    def _run_query(self, query: str, run_manager: Optional[CallbackManagerForChainRun] = None) -> SQLResult:
        m = _query_re.match(query.strip())
        query = m.group('query')

        try:
            sql_result = self.db.run(query, hard_limit=10)
            error = False
        except OperationalError as e:
            sql_result = e._message()
            error = True
        sql_result = sql_result or 'No results.'
        sql_result = f'```{sql_result}```'
        self.print_msgs([f'SQLResult: {sql_result}'], run_manager)
        return SQLResult(sql_result=sql_result, sql_error=error)

    def _get_answer(self, attempt: AIAttempt, result: SQLResult, run_manager: Optional[CallbackManagerForChainRun] = None) -> str:
        answer_prompt = SystemMessage(content=prompt.ANSWER_PROMPT.format())
        ai_msg = AIMessage(
            content=(
                f'StepByStep: [ ... ]\n'
                f'SQLQuery: {attempt.sql_query}\n'
                f'SQLResult: {result.sql_result}\n'
                f'Answer: '
            )
        )
        ai_response = self.llm.predict_messages(
            messages=[answer_prompt, attempt.human_message, ai_msg]
        )

        self.print_msgs([ai_response], run_manager)

        steps = separate_steps(ai_response.content)
        answer = steps.get('Answer') or ai_response.content

        return answer

    def _try_to_answer(self, user_prompt: str, max_attempts: int = 3, run_manager: Optional[CallbackManagerForChainRun] = None) -> Optional[str]:
        previous_attempts: list[FailedAttempt] = []
        for i in range(max_attempts):
            query_attempt = self._generate_query(user_prompt, previous_attempts, run_manager)
            if query_attempt.sql_query is None: return query_attempt.answer or query_attempt.full_content
            result = self._run_query(query_attempt.sql_query, run_manager)
            if i < max_attempts - 1 and result.sql_error:
                previous_attempts.append(FailedAttempt(attempt=query_attempt, result=result))
                continue
            else:
                return self._get_answer(query_attempt, result, run_manager)

    def run_sql(self, sql_query: str) -> str:
        l = sql_query.lower()
        restricted = ['delete', 'update', 'insert', 'create', 'alter', 'drop', 'pragma']
        if any(r in l for r in restricted):
            return 'Sorry, I can only answer SELECT queries.'
        result = self._run_query(sql_query)
        return result.sql_result

    def _call(self,
              inputs: dict[str, Any],
              run_manager: Optional[CallbackManagerForChainRun] = None):
        user_prompt = inputs['prompt']
        answer = self._try_to_answer(user_prompt, 1, run_manager)
        if answer is None:
            return {'response': 'Sorry, I failed to get an answer.'}
        else:
            return {'response': answer}
