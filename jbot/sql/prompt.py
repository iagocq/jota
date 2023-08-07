from langchain.prompts import PromptTemplate

DATABASE_DESCRIPTION_COURSES = """This database manages academic information for the university, covering courses, subjects, professors, class schedules, and classroom details, all interconnected to provide a comprehensive overview of the educational offerings.

Table Cursos:
\tid_curso VARCHAR(8) PK
\tnome_curso VARCHAR(255)
Example data for table Cursos:
id_curso\tnome_curso
G010\tCiência da Computação (Bacharelado)
G014\tSistemas de Informação (Bacharelado)
...

Table Disciplinas:
\tid_disc VARCHAR(8) PK
\tnome_disc VARCHAR(255)
Example data for table Disciplinas:
id_disc\tnome_disc\tcreditos
GCC125\tRedes de Computadores\t4
GAC106\tFundamentos de Programação I\t4
...

Table DisciplinasMatriz:
\tid_disc VARCHAR(8) PK
\tid_curso VARCHAR(8) PK
\tperiodo INT (nullable)
\tcat_eletiva VARCHAR(255) (nullable)
Example data for table DisciplinasMatriz:
id_disc\tid_curso\tperiodo\tcat_eletiva
GCC125\tG010\t5\tNULL
GCC176\tG010\tNULL\tA
GCC176\tG014\t4\tNULL
...

Table OfertasDisciplina:
\tid_oferta INT PK
\tid_disc VARCHAR(8)
\tid_curso VARCHAR(5)
\tprofessor VARCHAR(255)
Example data for table OfertasDisciplina:
id_oferta\tid_disc\tid_curso\tprofessor\tturma
1\tGCC125\tG010\tX\t10A
2\tGCC125\tG014\tX\t14A
...

Table Aulas:
\tid_oferta INT
\tlocal_curto VARCHAR(16)
\tlocal_completo VARCHAR(255)
\tdia_semana VARCHAR(8)
\thora_inicio INT
\thora_fim INT
Example data for table Aulas:
id_oferta\tlocal_curto\tlocal_completo\tdia_semana\thora_inicio\thora_fim
1\tPV2-201\tX\tquarta\t13\t15
1\tPV2-201\tX\tquinta\t13\t15
...
"""

_admin_prefix = """You're a database administrator. You are very cautious and attentive to the process you always follow to answer user prompts.

{database_description}
"""

step_descriptions = {
  "Input": "reword the user prompt. Write in your own words about the information that the user wants. List all variables in the question.",
  "TablesColumns": "write down the relevant tables and columns that have the desired information that the user wants.",
  "IntermediaryTablesColumns": "write down tables that are needed or should be joined in the query to answer the prompt.",
  "HaveAllInformation": "reason about the user's prompt and what information is available in the database. Can all the information be extracted from the database? If not, explain that more information is needed and stop the process.",
  "SQLPlan": "create a high level plan for the SQL query you're about to construct. Describe how each table should interact with each other and if there should be subqueries.",
  "UseLike": "respond with \"When comparing text, I'll do a text comparison using LIKE and %\"",
  "SQLQuery": 'based on your answers to "Input", "TablesColumns", "IntermediaryTablesColumns", and the database description, write a single sqlite query that will satisfy the user\'s prompt. Always limit the results to 5 if not specified by the user. Only write the SQL query. Only write one SQL query.',
  "DoesItLimit": 'focus solely on the query. Can there be too many results to the query? Does the query numerically limit the number of results? Explain your reasoning. If there can be too many results, report that and stop the process.',
  "SQLReasoning": 'reason about the query given in the SQLPlan step. On its own, explain what the query does and its limitations.',
  "IsItSpecific": "Answer the following question: how generic is the query? A query is generic when it doesn't take into account details of the user prompt that would narrow down the results. Then answer this: is the query specific to the user's prompt? If not, report that and stop the process.",
  "SQLResponse": "Place the response from the SQL query here.",
  "Answer": "Examine SQLResponse and answer the user's prompt directly. If SQLResponse is invalid, report that.",
  "Conclusion": "Explain what was done in at most 3 sentences. Report if any error occurred. Describe what should be done next to obtain more information if something is missing.",
  "Next": 'answer "done" or "retry"',
}

step_descriptions_2 = {
  "Reasoning": "reason about the user prompt and what it expects as an answer.",
  "TablesColumns": "write down the relevant tables and columns that have the desired information that the user wants.",
  "HaveAllInformation": "do you have all the information needed to build a SQL query that answers the prompt? If not, explain that more information is needed and stop the process.",
  "SQLQuery": 'write a single sqlite query that will satisfy the user\'s prompt. Only write the SQL query. Always use LIKE and % when comparing.',
  "SQLResponse": "Place the response from the SQL query here.",
  "Answer": "Examine SQLResponse and answer the user's prompt directly. If SQLResponse is invalid, report that.",
  "Next": 'answer "done" or "retry"',
}

step_answers = {
  "Input": '[answer here]',
  "TablesColumns": '[answer here]',
  "IntermediaryTablesColumns": '[answer here]',
  "HaveAllInformation": '[answer here]',
  "SQLPlan": '[answer here]',
  "UseLike": '[always answer with "When comparing text, I\'ll do a text comparison using LIKE and %"]',
  "SQLQuery": '[answer here]',
  "DoesItLimit": '[answer here]',
  "SQLReasoning": '[answer here]',
  "IsItSpecific": '[answer here]',
  "SQLResponse": '[answer here]',
  "Answer": '[answer here]',
  "Conclusion": '[answer here]',
  "Next": '[answer "done" or "retry"]',
  "Reasoning": '[answer here]',
}

def make_steps(steps: list[str], infix: str, descriptions: dict[str, str] = step_descriptions) -> str:
  descriptions = '\n\n'.join(f'{step}: {descriptions[step]}' for step in steps)
  answers = '\n'.join(f'{step}: {step_answers[step]}' for step in steps)
  return descriptions + infix + answers

_process_prefix = """Your job is to reason about prompts a user gives to you in plain text.

When a user provides you with a prompt, always do the following process:
"""

_process_infix = """
Repeat this entire process as many times as necessary

When providing your answers for the process, reply as following:
"""

_process_suffix = """
Very important:
- Do each step with caution and attention. A mistake in any step may carry on to later steps if not careful. Please be attentive!
- If you answer negatively to any question at any point, stop the process immediately, inform the user about what happenend, and start again!

You can stop at any point in the process by writing this after the step you just concluded:
Answer: [answer here]
Next: [answer "retry"]
"""

_retry_infix = """
These are your previous attempts:
{attempts}

Try again from the beginning:
"""

query_steps = [
  "Input",
  "TablesColumns",
  "IntermediaryTablesColumns",
  "HaveAllInformation",
  "SQLPlan",
  "UseLike",
  "SQLQuery",
  "DoesItLimit",
  "SQLReasoning",
  "IsItSpecific",
  "SQLResponse",
  "Answer",
  "Conclusion",
  "Next",
]

GEN_QUERY_PROMPT = PromptTemplate(
  input_variables=["database_description"],
  template=
    _admin_prefix +
    _process_prefix +
    make_steps(query_steps, _process_infix) +
    _process_suffix
)

query_steps_2 = [
  "Reasoning",
  "TablesColumns",
  "HaveAllInformation",
  "SQLQuery",
  "SQLResponse",
  "Answer",
  "Next",
]

GEN_QUERY_PROMPT_2 = PromptTemplate(
  input_variables=["database_description"],
  template=
    _admin_prefix +
    _process_prefix +
    make_steps(query_steps_2, _process_infix, step_descriptions_2) +
    _process_suffix
)

retry_steps = [
  "Input",
  "SQLQuery",
  "SQLResponse",
  "Answer",
  "Conclusion",
  "Next",
]

RETRY_PROMPT = PromptTemplate(
  input_variables=["attempts", "database_description"],
  template=
    _admin_prefix +
    _process_prefix +
    make_steps(retry_steps, _process_infix) +
    _retry_infix +
    _process_suffix
)

ai_steps = [
  "Input",
  "TablesColumns",
  "IntermediaryTablesColumns",
  "HaveAllInformation",
  "SQLPlan",
  "UseLike",
  "SQLQuery",
  "DoesItLimit",
  "SQLReasoning",
  "IsItSpecific",
]

ai_steps_2 = [
  "TablesColumns",
  "HaveAllInformation",
  "SQLQuery",
]

answer_steps = [
  "Input",
  "SQLPlan",
  "SQLQuery",
  "SQLResponse",
  "Answer",
  "Conclusion",
  "Next",
]

ANSWER_PROMPT = PromptTemplate(
  input_variables=[],
  template=
    _process_prefix +
    make_steps(
      answer_steps,
      _process_infix
    ) + _process_suffix
)

answer_steps_2 = [
  "SQLQuery",
  "SQLResponse",
  "Answer",
  "Next",
]

ANSWER_PROMPT_2 = PromptTemplate(
  input_variables=[],
  template=
    _process_prefix +
    make_steps(
      answer_steps_2,
      _process_infix,
      step_descriptions_2
    ) + _process_suffix
)