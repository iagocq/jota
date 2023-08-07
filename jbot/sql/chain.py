from langchain.chains.base import Chain
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from typing import Any, Optional
from . import prompt
from .db import SQLDatabase
import sqlite3

def separate_steps(all_steps: list[str], message: str):
    parts = {}
    last_idx = 0
    for prev, step in zip([None] + all_steps, all_steps):
        idx = message.find(f'{step}:')
        if prev is not None:
            part = message[last_idx+len(f'{prev}:'):idx].strip()
            parts[prev] = part
        last_idx = idx
    parts[step] = message[last_idx+len(f'{step}:'):].strip()
    return parts

class SQLChain(Chain):
    llm: BaseLanguageModel
    db: SQLDatabase
    database_description: str
    output_key: str = "sql_analysis"

    @property
    def input_keys(self) -> list[str]:
      return ["prompt"]

    @property
    def output_keys(self) -> list[str]:
      return [self.output_key]
  
    def _call(self,
              inputs: dict[str, Any],
              run_manager: Optional[CallbackManagerForChainRun] = None):
        gen_query_prompt = SystemMessage(
            content=prompt.GEN_QUERY_PROMPT_2.format(database_description=self.database_description)
        )

        run_manager.on_text(gen_query_prompt.content, color="blue", verbose=self.verbose)

        user_prompt = HumanMessage(content=f'{inputs["prompt"]}\n')
        run_manager.on_text(user_prompt.content, color="yellow", verbose=self.verbose)

        ai_response = self.llm.predict_messages(
            messages=[gen_query_prompt, user_prompt],
            stop=["\nSQLResponse:"]
        )
        run_manager.on_text(ai_response.content, color="green", verbose=self.verbose)

        steps = separate_steps(prompt.ai_steps_2, ai_response.content)
        query = steps['SQLQuery']

        try:
            response = self.db.run(query, hard_limit=5)
        except sqlite3.OperationalError as e:
            print(e.sqlite_errorname)
            response = str(e)

        run_manager.on_text(f'SQLResponse: {response!r}', color="red", verbose=self.verbose)
        # answer_prompt = SystemMessage(
        #     content=prompt.ANSWER_PROMPT_2.format()
        # )

        # run_manager.on_text(answer_prompt.content, color="blue", verbose=self.verbose)

        # ai_message = AIMessage(
        #     content=f'{ai_response.content}\nSQLResponse: `{response!r}`\n'
        # )

        # run_manager.on_text(ai_message.content, color="green", verbose=self.verbose)

        # response = self.llm.predict_messages([answer_prompt, ai_message])
        # run_manager.on_text(response.content, color="green", verbose=self.verbose)

        return {'sql_analysis': 'a'}

__all__ = [
  "SQLChain"
]
