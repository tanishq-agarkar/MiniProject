# 🧠 Emotion Detection Web App (RoBERTa)

A deep learning-based **multi-label emotion classification app** built using **RoBERTa** and deployed with **Streamlit**.
This app allows users to input text and detect multiple emotions simultaneously from 28 predefined categories.
---

## 📌 Features

* 🔍 Multi-label emotion detection (28 emotions)
* ⚡ Powered by RoBERTa transformer model
* 🧠 Handles noisy and real-world text
* 🎯 Threshold-based prediction
* 🌐 Interactive web app using Streamlit
* 📊 Optional visualization of emotion scores

---

## 🧾 Emotion Labels

```
admiration, amusement, anger, annoyance, approval, caring,
confusion, curiosity, desire, disappointment, disapproval,
disgust, embarrassment, excitement, fear, gratitude, grief,
joy, love, nervousness, optimism, pride, realization, relief,
remorse, sadness, surprise, neutral
```

---

## 🏗️ Project Structure

```
emotion-app/
│
├── app.py                # Streamlit web app
├── best_model.pt         # Trained model weights
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── (optional) train.py   # Training script
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/emotion-app.git
cd emotion-app
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the App Locally

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## ☁️ Deployment (Hugging Face Spaces)

1. Go to https://huggingface.co/spaces
2. Click **Create Space**
3. Select:

   * SDK: **Streamlit**
4. Upload:

   * `app.py`
   * `requirements.txt`
   * `best_model.pt`

---

## 🧠 Model Details

* **Base Model:** RoBERTa (`roberta-base`)
* **Task:** Multi-label text classification
* **Loss Function:** BCEWithLogits / Focal Loss
* **Max Length:** 160 tokens
* **Optimizer:** AdamW
* **Scheduler:** Linear warmup

---

## 📊 Evaluation Metrics

* **Primary Metric:** Macro F1 Score
* **Additional:**

  * Confusion Matrix
  * Classification Report

---

## 🎯 Example

**Input:**

```
I can't believe this happened, I'm so happy!
```

**Output:**

```
joy: 0.91  
surprise: 0.76
```

---

## 🔧 Customization

* Adjust prediction threshold in `app.py`:

```
threshold = 0.3
```

* Swap model:

```
roberta-base → deberta-v3-base
```

* Add UI improvements:

  * Emotion bar charts
  * Top-K predictions
  * Color-coded outputs

---

## ⚠️ Notes

* Large model files (>100MB) should be uploaded using **Git LFS** or hosted via model hubs
* Ensure correct PyTorch version compatibility
* CPU inference may be slower

---

## 📦 Requirements

```
torch
transformers
streamlit
numpy
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

* Open issues
* Submit pull requests
* Suggest improvements

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* Hugging Face Transformers
* GoEmotions Dataset
* Streamlit

---

## 👨‍💻 Author

**Your Name**
GitHub: https://github.com/your-username

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
