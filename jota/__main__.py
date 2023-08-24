import os
import dotenv
import asyncio
from .openai.api import OpenAIClient, UsageContext
from .openai.ai import OpenAIModel
from .agent import ConversationalAgent, History, ChatMessage, ClassificationAgent, QueryGeneratorAgent, HintMessage

dotenv.load_dotenv()

async def amain():
    email = os.getenv('EMAIL', '<empty>')
    schema = os.getenv('SCHEMA', '<empty>')
    client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY', 'x'))
    usage_context = UsageContext()
    model_3_5 = OpenAIModel(model='gpt-3.5-turbo', openai_client=client, usage_context=usage_context)
    model_4 = OpenAIModel(model='gpt-4', openai_client=client, usage_context=usage_context)

    hist = History(messages=[])
    bot = ConversationalAgent(
        history=hist,
        information=[
            f"The name of your creator is Iago. His email is {email}",
            "Your source code is available at https://github.com/iagocq/jota",
            "You have access to information related to the courses database."
            "You only provide factual information"
        ],
        model=model_3_5
    )

    classifier = ClassificationAgent(
        model=model_3_5,
        categories={
            'courses': 'Questions related to the courses database. This database contains information about the current courses, classes, classrooms, teachers.',
            'university': 'Questions related to the university that are not related to the courses database. This category encompasses historical university data, university regulations, high level description of courses.',
            'general': 'General conversation questions that do not fall under other categories.'
        }
    )

    sql = QueryGeneratorAgent(
        model=model_4,
        engine='sqlite',
        db_schema=schema,
        history=hist
    )

    name = input('Your name: ')
    while True:
        message = input('You: ')
        if message == 'quit':
            break
        if message == 'hist':
            print(hist)
            continue

        msg = ChatMessage(sender=name, content=message)
        classification = await classifier.classify(message)
        hint_msg = None
        if classification != 'general': print(f'Classification: {classification}')
        if classification == 'courses':
            # query = await sql.generate(message)
            # print(query)
            resp = f"""
There were no results for the user question
"""
            hint_msg = HintMessage(sender=None, content=resp)

        print(f'Bot: ', end='', flush=True)
        async for reply in bot.streaming_reply_to_message(msg, hint_msg):
            print(reply, end='', flush=True)
        print()

asyncio.run(amain())
