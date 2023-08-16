from flask import Flask, request, jsonify
from jbot.main import create_chain

chain = create_chain()
app = Flask(__name__)

@app.post('/prompt')
def prompt():
    json = request.get_json()
    prompt = json['prompt']
    context = json['context']
    full_prompt = (
        f'Context (ignore if not relevant to the prompt): """{context}"""\n'
        f'Prompt: """{prompt}"""'
    )
    answer = chain.run(full_prompt)
    return jsonify({'answer': answer})

@app.post('/query')
def query():
    json = request.get_json()
    results = chain.run_sql(json['query'])
    return jsonify({'results': results})
