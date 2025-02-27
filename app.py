from flask import Flask, render_template, request
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model("models/sentiment_analysis_lstm.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    return text

# Function to predict sentiment
def predict_sentiment(texts):
    processed_texts = [preprocess_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=500)
    predictions = model.predict(padded_sequences)
    return ["Positive" if p > 0.5 else "Negative" for p in predictions]

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    df = pd.read_csv(file)
    if "text" not in df.columns:
        return "CSV must have a 'text' column", 400

    df["Sentiment"] = predict_sentiment(df["text"])

    # Save the analyzed results to a CSV file
    output_path = "static/analyzed_results.csv"
    df.to_csv(output_path, index=False)

    # Count positive and negative tweets
    sentiment_counts = df["Sentiment"].value_counts()

    # Plot bar chart with dynamically assigned colors
    plt.figure(figsize=(6, 4))
    colors = ["green" if sentiment == "Positive" else "red" for sentiment in sentiment_counts.index]
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Tweets")
    plt.title("Sentiment Analysis Result")

    # Save the plot
    plot_path = "static/sentiment_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return render_template("result.html", plot_path=plot_path, sentiment_counts=sentiment_counts.to_dict(), download_path=output_path)


from flask import send_file

# Route to download the analyzed results as a CSV
@app.route("/download")
def download():
    output_path = "static/analyzed_results.csv"
    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
