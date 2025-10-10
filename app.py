from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import io
import sys
from ml_tools import get_bracket, bracket_predict

app = Flask(__name__)

# Load model and data on startup
model_name = 'final_iter'
MODEL = joblib.load(f'models/{model_name}.pkl')
STAT_DATA = pd.read_csv('data/player_profiles.csv', index_col=0)

# Tournament configurations
TOURNAMENTS = {
    'wimbledon_2023': {
        'name': 'Wimbledon 2023',
        'surface': 'Grass',
        'num_players': 128,
        'html_file': 'data/wimbledon_2023.html'
    }
}

@app.route('/')
def index():
    return render_template('index.html', tournaments=TOURNAMENTS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tournament_id = request.json.get('tournament_id')
        
        if tournament_id not in TOURNAMENTS:
            return jsonify({'error': 'Invalid tournament'}), 400
        
        tournament = TOURNAMENTS[tournament_id]
        
        # Get bracket from HTML file
        print("Getting bracket...")
        players = get_bracket(STAT_DATA)
        
        if not players:
            return jsonify({'error': 'No players found in bracket'}), 400
        
        # Capture the bracket prediction output
        output_bracket = []
        
        # Capture console output for display
        print(f"Running bracket prediction [using model: {model_name}]...")
        
        # Run bracket prediction
        bracket_predict(
            players=players,
            surface=tournament['surface'],
            num_players=tournament['num_players'],
            stat_data=STAT_DATA,
            model=MODEL,
            output_bracket=output_bracket,
            mode='store'
        )
        
        return jsonify({
            'success': True,
            'tournament': tournament['name'],
            'initial_players': players,  # Include the starting 128 players
            'output': output_bracket,
            'num_players': len(players)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)


