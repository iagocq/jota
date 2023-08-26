import os
import dotenv
import asyncio
from .openai.api import OpenAIClient, UsageContext
from .openai.ai import OpenAIModel
from .agent import ConversationalAgent, HistoryView, ChatMessage, Classifier, Hinter
from .sql_hinter import HintMessage, SQLHinter, EnhancerStep, SQLiteExecutor
import aiosqlite

dotenv.load_dotenv()

async def amain():
    email = os.getenv('EMAIL', '<empty>')
    schema = os.getenv('SCHEMA', '<empty>')
    client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY', 'x'))
    usage_context = UsageContext()
    model_3_5 = OpenAIModel(model='gpt-3.5-turbo', openai_client=client, usage_context=usage_context)
    model_4 = OpenAIModel(model='gpt-4', openai_client=client, usage_context=usage_context)

    hist = HistoryView()

    classifier = Classifier(
        model=model_3_5,
        categories={
            'courses': 'Questions related to the courses, classes, classrooms, and professors database.',
            'university': 'Questions related to the university that are not related to the courses database. This category encompasses historical university data, university regulations, high level description of courses.',
            'general': 'General conversation questions that do not fall under other categories.'
        }
    )

    enhances: list[EnhancerStep] = [
        EnhancerStep(model=model_3_5.override(temperature=0.5), n_generations=2),
        EnhancerStep(model=model_3_5.override(temperature=1), n_generations=5),
    ]

    db = await aiosqlite.connect('db.sqlite3')

    executor = SQLiteExecutor(engine='sqlite', db_schema=schema, connection=db)

    sql = SQLHinter(
        sql_executor=executor,
        enhancement_steps=enhances,
        history_view=hist
    )

    bot = ConversationalAgent(
        history_view=hist,
        information=[
            f"The name of your creator is Iago. His email is {email}",
            "Your source code is available at https://github.com/iagocq/jota",
            "You have access to information related to the courses database."
            "You try to provide factual information, but you may make mistakes."
        ],
        model=model_3_5,
        classifier=classifier,
        hinter=Hinter(
            generators={
                'courses': sql.hint
            }
        )
    )

    try:
        name = input('Your name: ')
        while True:
            message = input('You: ')
            if message == 'quit':
                break
            if message == 'hist':
                print(hist)
                continue

            msg = ChatMessage(role='user', sender=name, content=message)

            print(f'Bot: ', end='', flush=True)
            async for reply in bot.streaming_reply_to_message(msg):
                print(reply, end='', flush=True)
            print()
    finally:
        await db.close()

asyncio.run(amain())
