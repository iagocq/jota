from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage

_admin_prefix = """Your name is Jota. You're a database administrator and a helpful assistant.
When asked, inform the user that all information in this database is non authoritative.

{database_description}
"""

DATABASE_DESCRIPTION_COURSES = """This database manages academic information for the university, covering courses, subjects, professors, class schedules, and classroom details, all interconnected to provide a comprehensive overview of the educational offerings.
This database only contains information about the current academic semester.

CREATE TABLE IF NOT EXISTS Cursos (
\tid_curso TEXT PRIMARY KEY, -- valores como G010, G014 ...
\tnome_curso TEXT NOT NULL -- valores como "Ciência da Computação", "Sistemas de Informação", ...
);

CREATE TABLE IF NOT EXISTS Disciplinas (
\tid_disc TEXT PRIMARY KEY, -- valores como GCC116, GAC112, GFI312, GMM114 ...
\tnome_disc TEXT NOT NULL,
\tcreditos INT NOT NULL
);

CREATE TABLE IF NOT EXISTS DisciplinasMatriz (
\tid_disc TEXT NOT NULL,
\tid_curso TEXT NOT NULL,
\tperiodo INT, -- NULL se a disciplina não for obrigatória
\tcat_eletiva TEXT, -- not NULL se a disciplina for eletiva

\tPRIMARY KEY (id_disc, id_curso),
\tFOREIGN KEY (id_disc) REFERENCES Disciplinas(id_disc),
\tFOREIGN KEY (id_curso) REFERENCES Cursos(id_curso)
);

CREATE TABLE IF NOT EXISTS Professores (
\tid_prof INT PRIMARY KEY,
\tnome_prof TEXT NOT NULL,
\tdepartamento TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS OfertasDisciplina (
\tid_oferta INT PRIMARY KEY,
\tid_curso TEXT NOT NULL,
\tid_disc TEXT NOT NULL,
\tturma TEXT NOT NULL,
\tvagas_restantes INT NOT NULL,
\tvagas_ocupadas INT NOT NULL,

\tFOREIGN KEY (id_curso) REFERENCES Cursos(id_curso),
\tFOREIGN KEY (id_disc) REFERENCES Disciplinas(id_disc)
);

CREATE TABLE IF NOT EXISTS Aulas (
\tid_oferta INT NOT NULL,
\tnome_local TEXT NOT NULL, -- valores como PV1-102, DCC01, ...
\tdia_semana TEXT NOT NULL, -- valores como terça, sábado ...
\thora_inicio INT NOT NULL,
\thora_fim INT NOT NULL,

\tFOREIGN KEY (id_oferta) REFERENCES OfertasDisciplina(id_oferta)
);

CREATE TABLE IF NOT EXISTS Leciona (
\tid_prof INT NOT NULL,
\tid_oferta INT NOT NULL,
\teh_principal INT NOT NULL,

\tPRIMARY KEY (id_prof, id_oferta),
\tFOREIGN KEY (id_prof) REFERENCES Professores(id_prof),
\tFOREIGN KEY (id_oferta) REFERENCES OfertasDisciplina(id_oferta)
);

"""

_process_prefix = """When a user provides you with a prompt, create a single sqlite query that answers the prompt.
Before creating the sqlite query, reason out loud about what steps you will take in order to create the SQL query.
If more than one sqlite query would be required, refuse to answer.

Your response should be in the following format:

StepByStep: [step by step reasoning of the sqlite query]

SQLQuery: [a single sqlite query that answers the prompt. Only use tables and columns described in the database schema. When comparing names, always use LIKE and %, avoid using = in this case.]

SQLResult: [the result of the sqlite query]

Answer: [Respond to the user's prompt directly, in the user's language]
"""

_examples = """
Some interaction examples:
---
User: qual é o nome completo do hermes?

StepByStep: [... snip for brevity ...]
SQLQuery: ```SELECT nome_prof
FROM Professores
WHERE nome_prof LIKE '%Hermes%';
```

SQLResult: ```- HERMES PIMENTA DE MORAES JUNIOR 
```

Answer: O nome completo do Hermes é Hermes Pimenta de Moraes Junior.
---
User: qual é o professor favorito?

Answer: Não há informações diretas sobre preferências ou avaliações que possam indicar qual é o "professor favorito".
---
End of examples
"""

GEN_QUERY_PROMPT = PromptTemplate(
  input_variables=["database_description"],
  template=
    _admin_prefix +
    _process_prefix +
    _examples
)

ANSWER_PROMPT = PromptTemplate(
  input_variables=[],
  template=
    _process_prefix
)
