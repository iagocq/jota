from langchain.chains.base import Chain
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import SystemMessage, AIMessage, HumanMessage, BaseMessage
from typing import Any, Optional
from . import prompt
from .db import SQLDatabase
import sqlite3
import re
from pydantic import BaseModel
from pprint import pprint
from sqlalchemy.exc import OperationalError

SYSTEM_COLOR = "blue"
USER_COLOR = "yellow"
AI_COLOR = "green"
SQL_COLOR = "red"

class Attempt(BaseModel):
    answers: dict[str, str]
    sql_response: Optional[str] = None
    retry: bool = False
    exit: bool = False

    def __str__(self) -> str:
        return f'Query: {self.answers["SQLQuery"]}\nResponse: {self.sql_response!r}\n'

_steps_re = re.compile(r'(\w+):\s+(.*?)(?=\n\w+:|$)', re.DOTALL)
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
            content = f'{msg}\n'

        elif isinstance(msg, BaseMessage):
            content = msg.content
            if isinstance(msg, SystemMessage):
                color = SYSTEM_COLOR
            elif isinstance(msg, HumanMessage):
                color = USER_COLOR
            elif isinstance(msg, AIMessage):
                color = AI_COLOR
            else:
                raise TypeError(f"Unknown message type {msg.__class__.__name__}")

        if run_manager is not None:
            run_manager.on_text(content, color=color, verbose=self.verbose)

    def _generate_query(self,
                        user_prompt: str,
                        previous_attempts: list[Attempt],
                        run_manager: Optional[CallbackManagerForChainRun] = None) -> tuple[HumanMessage, Attempt]:

        if previous_attempts:
            p = prompt.RETRY_PROMPT.format(
                attempts='\n\n'.join(str(p) for p in previous_attempts),
                database_description=self.database_description
            )
        else:
            p = prompt.GEN_QUERY_PROMPT.format(
                database_description=self.database_description
            )

        gen_query_prompt = SystemMessage(content=p)
        self.print_msg(gen_query_prompt, run_manager)

        u_prompt = HumanMessage(content=f'{user_prompt.strip()}\n')
        self.print_msg(u_prompt, run_manager)

        ai_response = self.llm.predict_messages(
            messages=[gen_query_prompt, u_prompt],
            stop=["\nSQLResponse:"]
        )
        self.print_msg(ai_response, run_manager)

        steps = separate_steps(ai_response.content)
        return u_prompt, Attempt(answers=steps)

    def _generate_and_run_query(self,
                                user_prompt: str,
                                previous_attempts: list[Attempt],
                                run_manager: Optional[CallbackManagerForChainRun] = None) -> tuple[HumanMessage, Attempt]:
        u_prompt, attempt = self._generate_query(user_prompt, previous_attempts, run_manager)
        query = attempt.answers.get('SQLQuery')
        if query is None:
            # TODO
            return u_prompt, attempt
        query = query.strip('`').lstrip('sql')
        attempt.answers['SQLQuery'] = query
        try:
            sql_response = self.db.run(query, hard_limit=5)
        except OperationalError as e:
            sql_response = e._message()
        attempt.sql_response = sql_response
        self.print_msg(f'SQLResponse: {sql_response}', run_manager)
        self.print_msg('-'*30, run_manager)

        return u_prompt, attempt

    def _what_next(self,
                   attempt: Attempt,
                   u_prompt: HumanMessage,
                   run_manager: Optional[CallbackManagerForChainRun] = None) -> Attempt:
        answer_prompt = SystemMessage(content=prompt.ANSWER_PROMPT.format())
        self.print_msg(answer_prompt, run_manager)

        self.print_msg(u_prompt, run_manager)

        ai_message = AIMessage(
            content=(
                f'SQLQuery: {attempt.answers["SQLQuery"]}\n'
                f'SQLResponse: {attempt.sql_response}\n'
            )
        )
        self.print_msg(ai_message, run_manager)

        ai_response = self.llm.predict_messages(
            [answer_prompt, u_prompt, ai_message],
        )
        self.print_msg(ai_response, run_manager)
        
        steps = separate_steps(ai_response.content)
        if 'Next' in steps:
            n = steps['Next'].lower()
            if 'done' in n:
                attempt.retry = False
            elif 'user' in n:
                attempt.exit = True
            elif 'retry' in n:
                attempt.retry = True
        else:
            attempt.exit = True

        return attempt

    def _call(self,
              inputs: dict[str, Any],
              run_manager: Optional[CallbackManagerForChainRun] = None):
        user_prompt = inputs['prompt']
        attempts: list[Attempt] = []
        last: Attempt
        for _ in range(3):
            u_prompt, attempt = self._generate_and_run_query(user_prompt, attempts, run_manager)
            last = attempt
            if attempt.answers.get('SQLQuery') is None:
                continue
            if attempt.exit:
                break
            attempt = self._what_next(attempt, u_prompt, run_manager)
            attempts.append(attempt)

            if not attempt.retry or attempt.exit:
                break

        return {'response': last.answers.get('Answer', '<no response>')}

__all__ = [
  "SQLChain"
]
