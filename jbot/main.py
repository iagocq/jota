import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
  OpenAIFunctionsAgent,
  AgentExecutor,
  create_sql_agent,
  AgentType
)
from langchain.callbacks import get_openai_callback
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from .sql.db import SQLDatabase
import langchain
from .sql.chain import SQLChain
from .sql.prompt import DATABASE_DESCRIPTION_COURSES
import sys

def create_chain():
  dotenv.load_dotenv()

  llm = ChatOpenAI(temperature=0.5, verbose=True)

  db = SQLDatabase.from_uri(f'sqlite:///db.sqlite3')

  sql_chain = SQLChain(llm=llm, db=db, database_description=DATABASE_DESCRIPTION_COURSES, verbose=True)
  return sql_chain
