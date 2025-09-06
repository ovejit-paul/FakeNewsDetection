# Fake News Detection Project (SDP-2)

## Files
- `train_model.py` : Train the ML model on your dataset (CSV with `text` and `label`).
- `app.py` : Desktop Tkinter application to load trained model and predict Fake/Real news with explanations.
- `requirements.txt` : Python dependencies.
- `demo_news.csv` : Small sample dataset for quick testing.

## Setup
```bash
pip install -r requirements.txt
```

## Train Model
```bash
python train_model.py --data demo_news.csv --out model_artifacts.pkl
```

## Run App
```bash
python app.py
```
Then load `model_artifacts.pkl` in the app, paste text, and click Predict.
