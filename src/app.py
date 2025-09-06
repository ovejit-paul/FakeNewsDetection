import tkinter as tk
import ttkbootstrap as ttk
from tkinter import scrolledtext, messagebox
import joblib
import os

# Load model artifacts
try:
    artifacts = joblib.load("../models/artifacts.pkl")
    vectorizer = artifacts["vectorizer"]
    model = artifacts["classifier"]
except Exception as e:
    raise RuntimeError(f"Error loading model artifacts: {e}")

# GUI
app = ttk.Window(themename="cyborg")
app.title("Fake News Detector")
app.geometry("800x600")

title_label = ttk.Label(app, text="📰 Fake News Detection", font=("Segoe UI", 20, "bold"))
title_label.pack(pady=20)

input_box = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=80, height=10, font=("Segoe UI", 12))
input_box.pack(pady=10)

def predict_news():
    text = input_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter news text!")
        return
    
    vec_text = vectorizer.transform([text])
    pred = model.predict(vec_text)[0]
    prob = model.predict_proba(vec_text)[0][pred] * 100

    result = "✅ Prediction: TRUE News" if pred == 1 else "❌ Prediction: FAKE News"
    explanation = (
        "✔ This text matches patterns found in trusted news sources."
        if pred == 1 else
        "⚠ This text shares similarities with fake news patterns."
    )

    result_box.config(state=tk.NORMAL)
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, f"{result}\nConfidence: {prob:.2f}%\n\n{explanation}")
    result_box.config(state=tk.DISABLED)

predict_btn = ttk.Button(app, text="Check News", command=predict_news, bootstyle="success")
predict_btn.pack(pady=10)

result_box = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=80, height=8, font=("Segoe UI", 12), state=tk.DISABLED)
result_box.pack(pady=10)

app.mainloop()
