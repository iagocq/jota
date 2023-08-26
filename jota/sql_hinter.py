from .agent import Template, HistoryView, Message, HintMessage
from .ai import AIModelMessage, AIModel
from pydantic import BaseModel
from typing import Optional, Any
import re
from abc import ABC, abstractmethod

THINKSTEPBYSTEP = (
    'ThinkStepByStep: '
    '[extract relevant table names, relevant given information, relevant columns.'
    ' Give a step by step reasoning of the parts that make up the query.'
    ']'
)

THINKSTEPBYSTEP_ENHANCE = (
    'ThinkStepByStep: '
    '[answer the previous questions, step by step.'
    ' Explain what should be done to fix the query, step by step.'
    ']'
)

SQLQUERY = (
    'SQLQuery: '
    '[write a single {engine} query that answers the prompt.'
    ' Only use tables and columns described in the database schema.'
    ' When comparing names, always use LIKE and %, avoid using = in this case.'
    ']'
)

GENERATE_QUERY = Template(template=
f'''Generate a single {{engine}} query to the database that answers the user's prompt.
Use the history messages to provide context to the query.
If more than one {{engine}} query would be required, refuse to answer.
If there is not enough information to generate a valid query, refuse to answer.
Distinguish between machine-readable IDs and human-readable names.

Provide your response in the following format:

{THINKSTEPBYSTEP}

{SQLQUERY}'''
)

ENHANCE_QUERY = Template(template=
f'''Examine the query, the user prompt and possible errors that occurred when executing the query.

Answer the following questions:
- Does the query compare machine-readable IDs only with columns meant for machine-readable IDs?
- Does the query compare human-readable names only with columns meant for human-readable names?
- Does the query compare human-readable names using LIKE and %?
- Does the query make good use of the information provided by the user?
- Is any information provided by the user missing from the query?

Fix the query if anything is wrong with it. Create a completely new query if necessary.

Provide your response in the following format:

{THINKSTEPBYSTEP_ENHANCE}

{SQLQUERY}'''
)

class SQLResult(BaseModel):
    query: str
    error: Optional[str] = None
    column_names: Optional[list[str]] = None
    rows: Optional[list[list[str]]] = None

class SQLExecutor(BaseModel, ABC):
    engine: str
    db_schema: str

    @abstractmethod
    async def execute(self, query: str) -> SQLResult: ...

import aiosqlite

class SQLiteExecutor(SQLExecutor):
    connection: aiosqlite.Connection

    class Config:
        arbitrary_types_allowed = True

    async def execute(self, query: str) -> SQLResult:
        result = SQLResult(query=query)
        try:
            async with self.connection.execute(query) as cursor:
                result.column_names = [description[0] for description in cursor.description]
                rows = await cursor.fetchall()
                result.rows = [[str(cell) for cell in row] for row in rows]

        except Exception as e:
            result.error = str(e)
        return result

class EnhancerStep(BaseModel):
    model: AIModel
    n_generations: int

class SQLHinter(BaseModel):
    sql_executor: SQLExecutor
    history_view: HistoryView
    enhancement_steps: list[EnhancerStep]
    generate_prompt: Template = GENERATE_QUERY
    enhance_prompt: Template = ENHANCE_QUERY
    limit_results: int = 10

    _steps_re = re.compile(r'(\w+):\s+(.*?)(?=\n\w+:|$)', re.DOTALL)
    def _separate_steps(self, message: str) -> dict[str, str]:
        parts = {}
        steps = self._steps_re.findall(message)
        for step, answer in steps:
            parts[step] = answer.strip()
        return parts

    _query_re = re.compile(r'^(```(sql(ite)?)?)?(?P<query>.*?)(```)?$', re.DOTALL | re.IGNORECASE)
    def _get_query(self, query: str) -> Optional[str]:
        match = self._query_re.match(query.strip())
        if not match: return None
        return match.group('query').strip()

    def _prepare_msgs(self, message: Message, context: HistoryView, *, last_result: Optional[SQLResult] = None) -> list[AIModelMessage]:
        msgs: list[AIModelMessage] = []
        template = self.generate_prompt if last_result is None else self.enhance_prompt

        prompt_msg = AIModelMessage(role='system', content=template.format(engine=self.sql_executor.engine))
        msgs.append(prompt_msg)

        schema_msg = AIModelMessage(role='system', name='database_schema', content=self.sql_executor.db_schema)
        msgs.append(schema_msg)

        if last_result is not None:
            query_msg = AIModelMessage(role='system', name='query', content=last_result.query)
            msgs.append(query_msg)

            error_txt = last_result.error or 'Query generated no errors.'
            error_msg = AIModelMessage(role='system', name='error', content=error_txt)
            msgs.append(error_msg)

        msgs.extend(context.messages)

        content = message.content.strip()
        user_msg = AIModelMessage(role='user', content=content)
        msgs.append(user_msg)

        return msgs

    async def _generate_queries(self, enhancer: EnhancerStep, message: Message, context: HistoryView, *, last_result: Optional[SQLResult]) -> list[str]:
        msgs = self._prepare_msgs(message, context, last_result=last_result)
        queries = await enhancer.model.generate_multiple(msgs, n=enhancer.n_generations)
        result = []
        for query in queries:
            steps = self._separate_steps(query)
            if 'SQLQuery' not in steps: continue
            query = self._get_query(steps['SQLQuery'])
            if query is None: continue
            result.append(query)
        return result

    async def _check_query(self, query: str) -> bool:
        l = query.lower()
        restricted = ['delete', 'update', 'insert', 'create', 'alter', 'drop', 'pragma']
        if any(r in l for r in restricted): return False
        return True

    async def _enhance_step(self, enhancer: EnhancerStep, message: Message, context: HistoryView, last_result: Optional[SQLResult]) -> Optional[SQLResult]:
        queries = await self._generate_queries(enhancer, message, context, last_result=last_result)

        queries_by_results: list[tuple[SQLResult, int]] = []
        for query in queries:
            if not await self._check_query(query): continue

            results = await self.sql_executor.execute(query)
            if not results.rows:
                queries_by_results.append((results, 0))
            else:
                queries_by_results.append((results, len(results.rows)))

        queries_by_results.sort(key=lambda x: x[1])
        if not queries_by_results: return None
        return queries_by_results[0][0]

    async def generate_enhanced_query(self, message: Message, context: HistoryView) -> Optional[SQLResult]:
        last_result = None
        for enhancer in self.enhancement_steps:
            result = await self._enhance_step(enhancer, message, context, last_result=last_result)
            if not result: continue
            last_result = result
        return last_result

    async def hint(self, message: Message, context: HistoryView) -> HintMessage:
        result = await self.generate_enhanced_query(message, context)
        text: str = ''
        if not result:
            text = 'I could not generate a query for this prompt.'
        elif result.column_names and result.rows is not None:
            columns = '\t'.join(result.column_names)
            if len(result.rows) > self.limit_results:
                omitted = len(result.rows) - self.limit_results
                result_rows = result.rows[:self.limit_results]
            else:
                omitted = 0
                result_rows = result.rows

            rows = '\n'.join('\t'.join(str(cell) for cell in row) for row in result_rows)
            if omitted > 0:
                rows += f'\n...\nInform the user that {omitted} results were omitted.'

            text = (
                'The user query returned the following results:\n\n'
                f'Column names: {columns}\n'
                f'Rows:\n{rows}'
            )
        elif result.error:
            text = 'I encounted an error while fetching the information for the user.\n'

        return HintMessage(
            role='system',
            name='search_result',
            content=text
        )
