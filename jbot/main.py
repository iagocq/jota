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

def main():
  dotenv.load_dotenv()
  langchain.debug = True

  llm = ChatOpenAI(temperature=0.9, verbose=True)

  db = SQLDatabase.from_uri(f'sqlite:///db.sqlite3')
  # toolkit = SQLDatabaseToolkit(db=db, llm=llm)

  # courses_agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
  sql_chain = SQLChain(llm=llm, db=db, database_description=DATABASE_DESCRIPTION_COURSES, verbose=True)
  def ask_about_courses(human_like_question: str):
    return sql_chain.run(human_like_question)

  with get_openai_callback() as cb:
    tools = [
      Tool(
        name="Courses-Human",
        description=(
          "A human answers a plain-text, human-like question about the university's current courses, discipline offerings, curricula, classes, professors, and related topics."
          "\nCopy and paste the question you want to ask the human here."
        ),
        func=ask_about_courses
      )
      # Tool(
      #   name="RestaurantMenu-DB",
      #   func=menu_chain.run,
      #   description=(
      #     "useful for when you need to answer questions about the university restaurant's menu.\n"
      #     "The query should be in the form of a question containing full context."
      #   )
      # ),
      # Tool(
      #   name="news",
      #   func=news_chain.run,
      #   description=(
      #     "useful for when you need to answer questions about the university news and current events.\n"
      #     "The query should be in the form of a question containing full context."
      #   )
      # )
    ]

    message = "You choose what human to use to answer your questions."
    prompt = OpenAIFunctionsAgent.create_prompt(SystemMessage(content=message))
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.run(sys.argv[1])
    print(cb)
