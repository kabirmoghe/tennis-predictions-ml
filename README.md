# 🎾 Predicting Tennis Match Outcomes with ML

**Exploring the "Moneyball" of Tennis — can performance statistics alone explain why the top players win?**

---

## Overview

This project applies **machine learning** to predict the outcome of ATP Tour matches based purely on **performance data**, not by any measure of player identity. By transforming 20 years of ATP match data (pre-2023) into **player-level statistical profiles**, the goal was to see whether win/loss results could be explained — and predicted — through **momentum, surface-specific performance, and match-level consistency**.

Essentially an analytical dive into the dynamics of competition:  
> “If we strip away names and reputations, do the numbers still tell us who wins?”

---

## Core Components

### 1. Data Engineering & Feature Design

- **Ingested 20+ years** of ATP Tour match data from Jeff Sackmann's tennis_atp dataset (thousands of matches).
- Aggregated and transformed raw match-by-match data into **player-level statistical profiles** (`player_profiles.csv`), tracking:
  - **Surface-specific performance**: Win %, break points saved/faced, serve/return stats for Clay, Grass, and Hard courts.
  - **Momentum metrics**: Rolling averages of recent form and consistency.
  - **Contextual factors**: Current ATP ranking, head-to-head records.
- Each match was reframed as a **feature-based prediction problem**:  
  ```Player A (X_A stats) vs Player B (X_B stats) → Who wins?```
- This abstraction allows the model to generalize beyond specific player identities.

### 2. Model Development

- Tested multiple **classification models** for match outcome prediction through iterative experimentation:
  - **Logistic Regression** – as a baseline interpretable model.
  - **Random Forest** – for capturing non-linear relationships and feature interactions.
- Performed **hyperparameter tuning** and **cross-validation** to optimize predictive accuracy.
- Final model saved as `final_iter.pkl` after multiple iterations of feature engineering and model refinement.
- Evaluated models on accuracy, precision, recall, and interpretability using historical ATP match data.

### 3. Interactive Bracket Predictions UI (Flask)

- Built a toy **Flask web application** (`app.py`) that brings the model to life.
- Users can simulate the **entire Wimbledon 2023 tournament bracket** (128 players) in real-time.
- The app loads pre-trained models (`models/final_iter.pkl`) and player performance profiles to predict match outcomes round-by-round.
- Features an interactive UI showing bracket progression, demonstrating how player-level performance metrics drive **probabilistic match predictions** in a real tournament structure.

---

## Running the Web App

To run the interactive bracket simulator locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

Then navigate to `http://localhost:5000` to simulate the Wimbledon 2023 bracket and see the model predictions in action.

---

## Key Insights

- Player **momentum, game win %, and surface specialization** were stronger predictors of victory than ranking alone.  
- Models could reliably predict **>70%** of outcomes using pure statistical profiles.

---

## Project Structure

```text
📂 tennis-ml/
 ┣ 📄 app.py                    # Flask web app for bracket simulation
 ┣ 📄 ml_tools.py               # Core model training, inference, and bracket logic
 ┣ 📂 models/                   # Trained models (final_iter.pkl, etc.)
 ┣ 📂 data/                     # ATP match data, player profiles, rankings
 ┣ 📂 notebooks/                # Exploratory data analysis & model experimentation
 ┣ 📂 templates/                # HTML templates for Flask app
 ┣ 📂 static/                   # CSS styling for web interface
 ┗ 📄 requirements.txt          # Python dependencies
```
