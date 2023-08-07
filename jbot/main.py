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

def main():
  dotenv.load_dotenv()

  llm = ChatOpenAI(temperature=0.5, verbose=True)

  db = SQLDatabase.from_uri(f'sqlite:///db.sqlite3')
  # toolkit = SQLDatabaseToolkit(db=db, llm=llm)

  # courses_agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
  sql_chain = SQLChain(llm=llm, db=db, database_description=DATABASE_DESCRIPTION_COURSES, verbose=True)
  with get_openai_callback() as cb:
    sql_chain.run("qual é o número total de professores que dão aulas para pelo menos 2 disciplinas diferentes?")
    print(cb)
  exit()

  tools = [
    Tool(
      name="Courses-Smart-DB",
      description=(
        "This tool can answer questions about the university's current courses, course offerings, curricula, classes, professors, and related topics."
        "\nThis tool is connected to the university's database."
        "\nPass the user's question verbatim to this tool."
      ),
      func=courses_agent.run
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

  message = "You're a helpful assistant for students of Universidade Federal de Lavras. You always reply in portuguese."
  prompt = OpenAIFunctionsAgent.create_prompt(SystemMessage(content=message))
  agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

  agent_executor.run("liste 5 ofertas e seus horários para a turma 10A")
