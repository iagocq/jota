import dotenv
from langchain.chains.sql_database import create_sql_query_chain
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
  OpenAIFunctionsAgent,
  AgentExecutor,
  initialize_agent,
)
from langchain.tools import Tool, tool
from langchain.schema import SystemMessage
from random import choice

dotenv.load_dotenv()
tools = [
  Tool(
    name="Courses-DB",
    func=courses_chain.run,
    description=""
  )
]

@tool("Courses Database")
def search_database(query: str) -> str:
  """useful for when you need to answer questions about the university courses.
  The query should be in the form of a question containing full context.
  """
  return ""

@tool("Restaurant Menu")
def query_menu(query: str) -> str:
  """useful for when you need to answer questions about the university restaurant's menu.
  The query should be in the form of a question containing full context.
  """
  return choice(("1. feijão\n2. arroz\n3. farofa", "macarrão\nalface\novo\nfrango"))

@tool("News")
def query_main_site(query: str) -> str:
  """useful for when you need to answer questions about the university news and current events.
  The query should be in the form of a question containing full context."""
  return ["https://ufla.br/noticias/novo-reitor"]

llm = ChatOpenAI(temperature=0)
tools = [search_database, query_menu, query_main_site]

# system_message = SystemMessage(content="You are very powerful assistant, but bad at calculating lengths of words.")
prompt = OpenAIFunctionsAgent.create_prompt()
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executor = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent_executor.run("?")
