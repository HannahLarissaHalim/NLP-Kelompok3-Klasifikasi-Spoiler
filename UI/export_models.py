import os, joblib, json

EXPORT_DIR = "/content/drive/MyDrive/NLP_Kelompok3/models_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

joblib.dump(best_svm, os.path.join(EXPORT_DIR, "svm_model.pkl"))
print("svm_model.pkl")

joblib.dump(best_rf, os.path.join(EXPORT_DIR, "rf_model.pkl"))
print("rf_model.pkl")

joblib.dump(best_xgb, os.path.join(EXPORT_DIR, "xgb_model.pkl"))
print("xgb_model.pkl")

# simpan metrik ML ke dict sementara
ml_metrics = {}
for model_name, r in results.items():
    ml_metrics[model_name] = {
        "auc_roc" : round(r["auc_roc"],     4),
        "f1_1" : round(r["f1_1"],        4),
        "f1_0" : round(r["f1_0"],        4),
        "f1_macro" : round(r["f1_macro"],    4),
        "recall_1" : round(r["recall_1"],    4),
        "recall_0" : round(r["recall_0"],    4),
        "precision_1" : round(r["precision_1"], 4),
        "precision_0" : round(r["precision_0"], 4),
    }

# simpan sementara ke Drive
with open(os.path.join(EXPORT_DIR, "_ml_metrics_tmp.json"), "w") as f:
    json.dump(ml_metrics, f)
print("_ml_metrics_tmp.json (sementara)")

import os, pickle, joblib, json
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

EXPORT_DIR = "/content/drive/MyDrive/NLP_Kelompok3/models_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# simpan model Keras
model.save(os.path.join(EXPORT_DIR, "lstm_model.keras"))
print("lstm_model.keras")

# simpan tokenizer
with open(os.path.join(EXPORT_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
print("tokenizer.pkl")

# simpan threshold optimal
with open(os.path.join(EXPORT_DIR, "lstm_threshold.txt"), "w") as f:
    f.write(str(round(float(best_threshold), 4)))
print(f"lstm_threshold.txt ({best_threshold:.4f})")

# hitung metrik BiLSTM di test set
y_pred_optimal = (y_prob >= best_threshold).astype(int)
lstm_metrics = {
    "BiLSTM": {
        "auc_roc" : round(roc_auc_score(y_test, y_prob), 4),
        "f1_1" : round(f1_score(y_test, y_pred_optimal, pos_label=1), 4),
        "f1_0" : round(f1_score(y_test, y_pred_optimal, pos_label=0), 4),
        "f1_macro" : round(f1_score(y_test, y_pred_optimal, average="macro"), 4),
        "recall_1" : round(recall_score(y_test, y_pred_optimal, pos_label=1), 4),
        "recall_0" : round(recall_score(y_test, y_pred_optimal, pos_label=0), 4),
        "precision_1" : round(precision_score(y_test, y_pred_optimal, pos_label=1), 4),
        "precision_0" : round(precision_score(y_test, y_pred_optimal, pos_label=0), 4),
        "threshold" : round(float(best_threshold), 4),
    }
}
with open(os.path.join(EXPORT_DIR, "_lstm_metrics_tmp.json"), "w") as f:
    json.dump(lstm_metrics, f)
print("_lstm_metrics_tmp.json (sementara)")


import os, json, shutil
import torch
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

EXPORT_DIR = "/content/drive/MyDrive/NLP_Kelompok3/models_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# simpan model dan tokenizer IndoBERT
bert_export_dir = os.path.join(EXPORT_DIR, "indobert")
trainer.save_model(bert_export_dir)
tokenizer.save_pretrained(bert_export_dir)
print(f"indobert/ (disimpan di {bert_export_dir})")

# simpan threshold
with open(os.path.join(EXPORT_DIR, "bert_threshold.txt"), "w") as f:
    f.write(str(round(float(best_threshold), 4)))
print(f"bert_threshold.txt ({best_threshold:.4f})")

# hitung metrik IndoBERT di test set
y_pred_optimal = (test_probs >= best_threshold).astype(int)
bert_metrics = {
    "IndoBERT": {
        "auc_roc" : round(roc_auc_score(y_test_arr, test_probs), 4),
        "f1_1" : round(f1_score(y_test_arr, y_pred_optimal, pos_label=1), 4),
        "f1_0" : round(f1_score(y_test_arr, y_pred_optimal, pos_label=0), 4),
        "f1_macro" : round(f1_score(y_test_arr, y_pred_optimal, average="macro"), 4),
        "recall_1" : round(recall_score(y_test_arr, y_pred_optimal, pos_label=1), 4),
        "recall_0" : round(recall_score(y_test_arr, y_pred_optimal, pos_label=0), 4),
        "precision_1" : round(precision_score(y_test_arr, y_pred_optimal, pos_label=1), 4),
        "precision_0" : round(precision_score(y_test_arr, y_pred_optimal, pos_label=0), 4),
        "threshold" : round(float(best_threshold), 4),
    }
}
with open(os.path.join(EXPORT_DIR, "_bert_metrics_tmp.json"), "w") as f:
    json.dump(bert_metrics, f)
print("_bert_metrics_tmp.json (sementara)")


import os, json

EXPORT_DIR = "/content/drive/MyDrive/NLP_Kelompok3/models_export"

all_metrics = {}

for tmp_file in ["_ml_metrics_tmp.json", "_lstm_metrics_tmp.json", "_bert_metrics_tmp.json"]:
    path = os.path.join(EXPORT_DIR, tmp_file)
    if os.path.exists(path):
        with open(path) as f:
            all_metrics.update(json.load(f))
        print(f"dimuat: {tmp_file}")
    else:
        print(f"tidak ditemukan: {tmp_file} — skip")

final = {"models": all_metrics}
with open(os.path.join(EXPORT_DIR, "metrics.json"), "w") as f:
    json.dump(final, f, indent=2)

print("\nmetrics.json berhasil dibuat!")
print(f"Model yang tercakup: {list(all_metrics.keys())}")
print("\nLangkah selanjutnya:")
print("1. Download folder models_export dari Google Drive")
print("2. Rename jadi 'models'")
print("3. Letakkan di dalam folder spoiler_detector/")
print("4. Jalankan: python3 app.py")
