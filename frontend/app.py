import os
from flask import Flask, render_template, request, jsonify, session
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.intake import run_intake_agent
from utils.llm_client import get_llm_client

app = Flask(__name__)
app.secret_key = "super_secret_hackathon_key"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    if 'history' not in session:
        session['history'] = []
        
    session['history'].append(user_message)
    full_context = " ".join(session['history'])
    
    try:
        client, model = get_llm_client()
        result = run_intake_agent(client, model, full_context)
        
        if result.get('clarification_needed'):
            bot_reply = result['clarification_needed']
            status = 'clarifying'
        else:
            bot_reply = "Excellent! I've built your structured risk profile from our conversation. Ready to proceed to the next Agent!"
            status = 'complete'
            
        return jsonify({
            'reply': bot_reply,
            'status': status,
            'profile': result if status == 'complete' else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    session.clear()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
