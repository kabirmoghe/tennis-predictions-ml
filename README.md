# Tennis Bracket Predictor

AI-powered tennis tournament bracket predictions using machine learning.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask app:**
   ```bash
   python app.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5000`

## Usage

1. Select a tournament from the dropdown (currently only Wimbledon 2023 is available)
2. Click "Generate Predictions"
3. View the AI-generated bracket predictions

## Features

- Clean, modern UI with gradient design
- Real-time bracket predictions
- Fuzzy name matching for player data
- Console output display showing match-by-match predictions

## Project Structure

```
tennis-ml/
├── app.py              # Flask application
├── ml_tools.py         # ML prediction functions
├── templates/          # HTML templates
│   └── index.html
├── static/            # CSS and static files
│   └── style.css
├── data/              # Tournament data and player stats
├── models/            # Trained ML models
└── requirements.txt   # Python dependencies
```