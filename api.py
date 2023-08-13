from flask import Flask, request, jsonify
from jbot.main import create_chain

chain = create_chain()

app = Flask(__name__)

@app.post('/prompt')
def prompt():
    json = request.get_json()
    answer = chain.run(json['prompt'])
    return jsonify({'answer': answer})

@app.post('/query')
def query():
    json = request.get_json()
    results = chain.run_sql(json['query'])
    return jsonify({'results': results})
