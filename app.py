"""
Spoiler Detector v2 — Backend Flask
Model: SVM, Random Forest, XGBoost (Pipeline + TF-IDF), BiLSTM (Keras), IndoBERT (HuggingFace)

Struktur folder models/ yang dibutuhkan:
  models/
    svm_model.pkl           <- best_svm (Pipeline: TF-IDF + LinearSVC)
    rf_model.pkl            <- best_rf  (Pipeline: TF-IDF + RF)
    xgb_model.pkl           <- best_xgb (Pipeline: TF-IDF + XGBoost)
    lstm_model.keras        <- model BiLSTM Keras
    tokenizer.pkl           <- Keras Tokenizer
    lstm_threshold.txt      <- satu baris angka: best_threshold BiLSTM
    indobert/               <- folder hasil trainer.save_model()
        config.json
        model.safetensors
        tokenizer_config.json
        ...
    bert_threshold.txt      <- satu baris angka: best_threshold IndoBERT
    metrics.json            <- hasil evaluasi semua model dari Colab

CARA EKSPOR -> lihat export_models_v2.py
"""

from flask import Flask, request, jsonify, render_template
import os, re, pickle, warnings
import numpy as np

warnings.filterwarnings("ignore")

app = Flask(__name__)

# CONFIG 
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MAXLEN_LSTM = 1773   # p90 review_text_clean
MAX_LEN_BERT = 256    # MAX_LEN notebook IndoBERT

# PREPROCESSING
def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def stem_text_for_ml(text: str, stemmer, stopwords_set: set) -> str:
    if not text:
        return ""
    WHITELIST = {"ternyata", "akhirnya", "tiba", "tiba-tiba"}
    tokens = [t for t in text.split() if t not in stopwords_set or t in WHITELIST]
    return stemmer.stem(" ".join(tokens))

# LAZY CACHE
_models = {}
_stemmer = None
_stopwords = None

def get_stopwords():
    global _stopwords
    if _stopwords is None:
        try:
            import nltk
            from nltk.corpus import stopwords as nltk_sw
            nltk.download("stopwords", quiet=True)
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
            WHITELIST = {"ternyata", "akhirnya", "tiba", "tiba-tiba"}
            _stopwords = (set(StopWordRemoverFactory().get_stop_words()) |
                          set(nltk_sw.words("indonesian"))) - WHITELIST
        except Exception as e:
            print(f"[WARN] stopwords gagal: {e}")
            _stopwords = set()
    return _stopwords

def get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            _stemmer = StemmerFactory().create_stemmer()
        except Exception as e:
            print(f"[WARN] stemmer gagal: {e}")
    return _stemmer

def load_pkl(name: str, filename: str):
    if name in _models:
        return _models[name]
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        import joblib
        obj = joblib.load(path)
        _models[name] = obj
        print(f"[OK] {name} dimuat")
        return obj
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return None

def load_lstm():
    if "lstm" in _models:
        return _models["lstm"]
    path = os.path.join(MODEL_DIR, "lstm_model.keras")
    if not os.path.exists(path):
        return None
    try:
        from tensorflow.keras.models import load_model as keras_load
        obj = keras_load(path)
        _models["lstm"] = obj
        print("[OK] BiLSTM dimuat")
        return obj
    except Exception as e:
        print(f"[ERROR] BiLSTM: {e}")
        return None

def load_bert():
    if "bert_model" in _models:
        return _models["bert_model"], _models["bert_tokenizer"]
    bert_dir = os.path.join(MODEL_DIR, "indobert")
    if not os.path.exists(bert_dir):
        return None, None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tok = AutoTokenizer.from_pretrained(bert_dir)
        model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
        model.eval()
        _models["bert_model"]     = model
        _models["bert_tokenizer"] = tok
        print("[OK] IndoBERT dimuat")
        return model, tok
    except Exception as e:
        print(f"[ERROR] IndoBERT: {e}")
        return None, None

def read_threshold(filename: str, default: float = 0.5) -> float:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        try:
            return float(open(path).read().strip())
        except:
            pass
    return default

# PREDIKSI 
def predict_ml(text_clean: str, model_key: str) -> dict:
    filemap = {"svm": "svm_model.pkl", "rf": "rf_model.pkl", "xgb": "xgb_model.pkl"}
    model = load_pkl(model_key, filemap[model_key])
    if model is None:
        return {"error": f"Model '{model_key.upper()}' belum tersedia di folder models/."}

    stemmer = get_stemmer()
    stopwords = get_stopwords()
    text_stem = stem_text_for_ml(text_clean, stemmer, stopwords) if stemmer else text_clean

    label = int(model.predict([text_stem])[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text_stem])[0]
        prob_spoiler = float(proba[1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function([text_stem])[0])
        prob_spoiler = 1 / (1 + np.exp(-score))
    else:
        prob_spoiler = 1.0 if label == 1 else 0.0

    return {
        "label" : label,
        "is_spoiler" : bool(label == 1),
        "prob_spoiler" : round(prob_spoiler * 100, 1),
        "prob_nonspoiler" : round((1 - prob_spoiler) * 100, 1),
        "text_processed" : text_stem[:200] + "..." if len(text_stem) > 200 else text_stem,
    }

def predict_lstm(text_clean: str) -> dict:
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        return {"error": "TensorFlow tidak terinstall."}

    tokenizer = load_pkl("tokenizer", "tokenizer.pkl")
    model = load_lstm()
    if tokenizer is None or model is None:
        return {"error": "Model BiLSTM atau tokenizer belum tersedia di folder models/."}

    seq = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=MAXLEN_LSTM, padding="post", truncating="post")
    prob_spoiler = float(model.predict(padded, verbose=0)[0][0])
    threshold = read_threshold("lstm_threshold.txt", 0.5)
    label = 1 if prob_spoiler >= threshold else 0

    return {
        "label" : label,
        "is_spoiler" : bool(label == 1),
        "prob_spoiler" : round(prob_spoiler * 100, 1),
        "prob_nonspoiler" : round((1 - prob_spoiler) * 100, 1),
        "text_processed" : text_clean[:200] + "..." if len(text_clean) > 200 else text_clean,
    }

import concurrent.futures

def _bert_inference(text_clean: str, model_dir: str, max_len: int) -> dict:
    import os, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    bert_dir = os.path.join(model_dir, "indobert")
    tok = AutoTokenizer.from_pretrained(bert_dir)
    model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
    model.eval()

    inputs = tok(
        text_clean,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    prob_spoiler = float(probs[1])

    # baca threshold langsung di sini
    threshold_path = os.path.join(model_dir, "bert_threshold.txt")
    threshold = 0.5
    if os.path.exists(threshold_path):
        try:
            threshold = float(open(threshold_path).read().strip())
        except:
            pass

    label = 1 if prob_spoiler >= threshold else 0

    return {
        "label": label,
        "is_spoiler": bool(label == 1),
        "prob_spoiler": round(prob_spoiler * 100, 1),
        "prob_nonspoiler": round((1 - prob_spoiler) * 100, 1),
        "text_processed": text_clean[:200] + "..." if len(text_clean) > 200 else text_clean,
    }

def predict_bert(text_clean: str) -> dict:
    try:
        import torch
    except ImportError:
        return {"error": "PyTorch tidak terinstall."}

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_bert_inference, text_clean, MODEL_DIR, MAX_LEN_BERT)
        try:
            return future.result(timeout=60)
        except Exception as e:
            return {"error": f"IndoBERT inference gagal: {str(e)}"}

# ROUTES 
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    model_choice = data.get("model", "svm").lower()

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong."}), 400
    if len(text) < 20:
        return jsonify({"error": "Teks terlalu pendek (minimal 20 karakter)."}), 400

    text_clean = clean_text(text)

    if model_choice in ("svm", "rf", "xgb"):
        result = predict_ml(text_clean, model_choice)
    elif model_choice == "lstm":
        result = predict_lstm(text_clean)
    elif model_choice == "bert":
        result = predict_bert(text_clean)
    else:
        return jsonify({"error": f"Model '{model_choice}' tidak dikenali."}), 400

    if "error" in result:
        return jsonify(result), 503

    result["input_length"] = len(text)
    result["clean_length"] = len(text_clean)
    result["model_used"]   = model_choice.upper()
    return jsonify(result)

@app.route("/status")
def status():
    checks = {
        "svm" : os.path.exists(os.path.join(MODEL_DIR, "svm_model.pkl")),
        "rf" : os.path.exists(os.path.join(MODEL_DIR, "rf_model.pkl")),
        "xgb" : os.path.exists(os.path.join(MODEL_DIR, "xgb_model.pkl")),
        "tokenizer": os.path.exists(os.path.join(MODEL_DIR, "tokenizer.pkl")),
        "lstm" : os.path.exists(os.path.join(MODEL_DIR, "lstm_model.keras")),
        "bert" : os.path.exists(os.path.join(MODEL_DIR, "indobert", "config.json")),
    }
    return jsonify({"models": checks})

@app.route("/metrics")
def metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "metrics.json belum tersedia."}), 404

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Spoiler Detector v2 — NLP Kelompok 3")
    print(f"Folder model : {MODEL_DIR}")
    print("Buka browser : http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=7860)
