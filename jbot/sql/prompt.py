from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage

_admin_prefix = """Your name is Jota. You're a portuguese-speaking database administrator and a helpful assistant.
Above all, you're a database administrator. You are very cautious and attentive to the process you always follow to answer user prompts.
When asked, inform the user that all information in this database is non authoritative.

{database_description}
"""

DATABASE_DESCRIPTION_COURSES = """This database manages academic information for the university, covering courses, subjects, professors, class schedules, and classroom details, all interconnected to provide a comprehensive overview of the educational offerings.
This database only contains information about the current academic semester.

Columns for table Cursos:
\tid_curso TEXT PK
\tnome_curso TEXT
Example data for table Cursos:
id_curso\tnome_curso
G010\tCiência da Computação (Bacharelado)
G014\tSistemas de Informação (Bacharelado)
...

Columns for table Disciplinas:
\tid_disc TEXT PK
\tnome_disc TEXT
Example data for table Disciplinas:
id_disc\tnome_disc
GCC125\tRedes de Computadores
GAC106\tFundamentos de Programação I
...

Columns for table DisciplinasMatriz:
\tid_disc TEXT PK
\tid_curso TEXT PK
\tperiodo INT (null for electives)
\tis_eletiva INT
Example data for table DisciplinasMatriz:
id_disc\tid_curso\tperiodo\tis_eletiva
GCC125\tG010\t5\t0
GCC176\tG010\tNULL\t1
GCC176\tG014\t4\t0
...

Columns for table Professores:
\tid_prof INT PK
\tnome_prof TEXT
Example data for table Professores:
id_prof\tnome_prof
1\tX da Silva
2\tY Guimarães
...

Columns for table OfertasDisciplina:
\tid_oferta INT PK
\tid_disc TEXT
\tid_curso TEXT
\tid_prof INT
\tturma TEXT
\tvagas_restantes INT
\tvagas_ocupadas INT
Example data for table OfertasDisciplina:
id_oferta\tid_disc\tid_curso\tid_prof\tturma\tvagas_restantes\tvagas_ocupadas
1\tGCC125\tG010\t1\t10A\t10\t30
2\tGCC125\tG014\t1\t14A\t10\t30
...

Columns for table AulasOferta:
\tid_oferta INT
\tid_disc TEXT
\tnome_local TEXT
\tdia_semana TEXT
\thora_inicio INT
\thora_fim INT
Example data for table AulasOferta:
id_oferta\tid_disc\tnome_local\tdia_semana\thora_inicio\thora_fim
1\tGCC125\tPV2-201\tquarta\t13\t15
1\tGCC125\tPV2-201\tquinta\t13\t15
...
"""

_process_prefix = """Your job is to reason about prompts a user gives to you in plain text.

When a user provides you with a prompt, always do the following process:
"""

step_descriptions = {
  "Reasoning": "reason about the user prompt and what it expects as an answer.",
  "HowToFix": "explain how to fix the problems from previous attempts. Explain what was missing from the other attempts or was wrongly assumed to be true.",
  "TablesColumns": "write down the relevant tables and columns that have the desired information that the user wants. Write about all tables that need to be joined to answer the query.",
  "HaveAllInformation": "does the database have all the information needed to answer the prompt? Do you have all the information needed to build a SQL query that answers the prompt? If not, ask for more information from the user and stop the process.",
  "SQLQuery": 'write a single sqlite query that will satisfy the user\'s prompt. Only write the SQL query. ALWAYS use LIKE and % for comparing text, never use = in this case. Use only tables and their columns described in the database schema.',
  "SQLResult": "Place the results from the SQL query here.",
  "SQLReasoning": 'reason about the query given in the SQLQuery step. On its own, explain what the query does and its limitations. Does the query answer the user\'s prompt? If not, explain why not. If it does, explain how it does.',
  "Answer": "Answer the user directly. Be mindful of limited results. Mention the number of omitted rows when relevant. If SQLResult is invalid, report that.",
  "Next": 'answer with just either "done", "ask for user input" or "retry with another query"',
  "Action": 'answer with either "successful query", "retry with another query" or "ask for more information"'
}

step_answers = {
  "UseLike": '[always answer with "When comparing text, I\'ll do a text comparison using LIKE and %"]',
  "Next": '[answer with just either "done", "ask for user input" or "retry with another query"]',
}

def make_steps(steps: list[str], descriptions: dict[str, str] = step_descriptions) -> str:
  descriptions_str = '\n'.join(f'{step}: {descriptions[step]}' for step in steps)
  return descriptions_str

_examples = """
Some interaction examples:
---
User: quem é o professor que ministra mais disciplinas para Engenharia Florestal?

Reasoning: The user wants to know the name of the professor that teaches the highest number of classes to the Engenharia Florestal course.
TablesColumns: The name of the professor is available in the Professores.nome_prof column. The disciplines that the professor teaches are available in the OfertasDisciplina.id_disc column. There should be an entry for Engenharia Florestal in the Cursos.nome_disc column. The following tables should be joined: Professores, OfertasDisciplina, Cursos.
HaveAllInformation: All the information is obtainable with the database.
SQLQuery: SELECT P.nome_prof, COUNT(DISTINCT OD.id_disc) as num_disciplinas FROM Professores P JOIN OfertasDisciplina OD ON P.id_prof = OD.id_prof JOIN Cursos C ON C.id_curso = OD.id_curso WHERE C.nome_curso LIKE '%Engenharia Florestal%' GROUP BY P.id_prof ORDER BY num_disciplinas DESC LIMIT 1;
SQLResult: ```- ANA CAROLINA MAIOLI CAMPOS BARBOSA\t4
```
Answer: O professor que ministra mais disciplinas para Engenharia Florestal é Ana Carolina Maioli Campos Barbosa, com um total de 4 disciplinas ministradas.
Action: successful query
---
User: qual professor é o mais favorito?

Reasoning: The user wants to know the name of the professor that is the most popular.
TablesColumns: The database does not have any information about the popularity of professors.
HaveAllInformation: The database does not have all the information needed to answer the prompt.
Answer: A base de dados não contém informações sobre a popularidade dos professores.
Action: ask for more information
---
End of examples
"""

_process_suffix = """
Very important:
- Do each step with caution and attention. A mistake in any step may carry on to later steps if not careful. Please be attentive!
- If you answer negatively to any question at any point, stop the process immediately, inform the user about what happenend, and start again!

You can stop at any point in the process by writing this after the step you just concluded:
Answer: [answer here]
Action: [answer here]
"""

_retry_infix = """

Here are SQL queries that failed to answer the user's prompt. Explain what went wrong.
{attempts}

Do the process from the beginning. Explain what went wrong.
"""

query_steps = [
  "Reasoning",
  "TablesColumns",
  "HaveAllInformation",
  "SQLQuery",
  "SQLResult",
  "Answer",
  "Action",
]

GEN_QUERY_PROMPT = PromptTemplate(
  input_variables=["database_description"],
  template=
    _admin_prefix +
    _process_prefix +
    make_steps(query_steps) +
    _examples +
    _process_suffix
)

retry_steps = [
  "Reasoning",
  "HowToFix",
  "TablesColumns",
  "HaveAllInformation",
  "SQLQuery",
  "SQLResult",
  "Answer",
  "Action",
]

retry_examples = [
  HumanMessage(content="quais disciplinas são ministradas pelo professor que dá aula no maior número de salas diferentes?"),
  AIMessage(content=
"""Reasoning: The user wants to know the disciplines that are taught by the professor who teaches in the highest number of different classrooms.
TablesColumns: The name of the professor is available in the Professores.nome_prof column. The classrooms where the professor teaches are available in the AulasOferta.nome_local column. The disciplines that the professor teaches are available in the OfertasDisciplina.id_disc column. The following tables should be joined: Professores, AulasOferta, OfertasDisciplina.
HaveAllInformation: All the information is obtainable with the database.
SQLQuery: SELECT P.nome_prof, OD.id_disc FROM Professores P JOIN OfertasDisciplina OD ON P.id_prof = OD.id_prof JOIN AulasOferta AO ON AO.id_oferta = OD.id_oferta GROUP BY P.id_prof HAVING COUNT(DISTINCT AO.nome_local) = (SELECT MAX(num_salas) FROM (SELECT COUNT(DISTINCT nome_local) as num_salas FROM AulasOferta GROUP BY id_prof) AS subquery);
SQLResult: (sqlite3.OperationalError) no such column: id_prof
HowToFix: There is no column named id_prof in the AulasOferta table.
"""
  )
]

RETRY_PROMPT = PromptTemplate(
  input_variables=["attempts", "database_description"],
  template=
    _admin_prefix +
    _process_prefix +
    make_steps(retry_steps) +
    _retry_infix +
    _process_suffix
)

ai_steps = [
  "Reasoning",
  "TablesColumns",
  "HaveAllInformation",
  "SQLQuery",
]

answer_examples = [
    HumanMessage(content="quem é o professor que ministra mais disciplinas para Engenharia Florestal?"),
    AIMessage(content=
"""SQLQuery: SELECT P.nome_prof, COUNT(DISTINCT OD.id_disc) as num_disciplinas FROM Professores P JOIN OfertasDisciplina OD ON P.id_prof = OD.id_prof JOIN Cursos C ON C.id_curso = OD.id_curso WHERE C.nome_curso LIKE '%Engenharia Florestal%' GROUP BY P.id_prof ORDER BY num_disciplinas DESC LIMIT 1;
SQLResult: `[("ANA CAROLINA MAIOLI CAMPOS BARBOSA", 4)]`
Answer: A professora que ministra mais disciplinas para Engenharia Florestal é Ana Carolina Maioli Campos Barbosa, com um total de 4 disciplinas ministradas.
Action: successful query"""),
]

answer_steps = [
  "SQLQuery",
  "SQLResult",
  "Answer",
  "Action",
]

ANSWER_PROMPT = PromptTemplate(
  input_variables=[],
  template=
    _process_prefix +
    make_steps(answer_steps,) +
    _process_suffix
)
