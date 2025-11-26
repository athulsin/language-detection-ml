from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

# -------------------------------------------------
# Load saved model and vectorizer
# -------------------------------------------------
model = joblib.load("lang_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# -------------------------------------------------
# Helper: clean text (same as in training)
# -------------------------------------------------
def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


# -------------------------------------------------
# Compute performance metrics once at startup
# -------------------------------------------------
def compute_performance():
    try:
        df = pd.read_csv("Language Detection.csv")

        # basic cleaning (same as in training script)
        df = df.drop_duplicates(subset=["Text"])
        df = df.dropna()
        df["Text"] = df["Text"].astype(str).str.lower()

        X = df["Text"]
        y = df["Language"]

        # You just need a test split to evaluate the *saved* model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        # Per-class rows for nice display in template
        class_rows = []
        for label in model.classes_:
            metrics = report.get(label)
            if not metrics:
                continue
            class_rows.append(
                {
                    "label": label,
                    "precision": round(metrics["precision"], 2),
                    "recall": round(metrics["recall"], 2),
                    "f1": round(metrics["f1-score"], 2),
                    "support": int(metrics["support"]),
                }
            )

        overall = {
            "accuracy": round(report["accuracy"], 3),
            "macro_precision": round(report["macro avg"]["precision"], 2),
            "macro_recall": round(report["macro avg"]["recall"], 2),
            "macro_f1": round(report["macro avg"]["f1-score"], 2),
            "weighted_precision": round(report["weighted avg"]["precision"], 2),
            "weighted_recall": round(report["weighted avg"]["recall"], 2),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 2),
        }

        return class_rows, overall

    except Exception as e:
        print("Error computing performance metrics:", e)
        return [], {}


performance_rows, overall_metrics = compute_performance()


# -------------------------------------------------
# Home page – prediction
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    predicted_language = None
    confidence = None
    top_probs = []
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("text", "")
        cleaned = preprocess_text(input_text)

        if cleaned.strip():
            X_vec = vectorizer.transform([cleaned])

            # Predicted label
            pred = model.predict(X_vec)[0]
            predicted_language = pred

            # Confidence + top probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_vec)[0]
                classes = model.classes_
                lang_probs = list(zip(classes, probs))
                # sort descending by probability
                lang_probs.sort(key=lambda x: x[1], reverse=True)

                best_lang, best_prob = lang_probs[0]
                confidence = round(best_prob * 100, 2)

                # top 5 for table
                top_probs = [
                    {"label": l, "prob": round(p * 100, 2)} for l, p in lang_probs[:5]
                ]

    return render_template(
        "index.html",
        input_text=input_text,
        predicted_language=predicted_language,
        confidence=confidence,
        top_probs=top_probs,
    )


# -------------------------------------------------
# Performance page
# -------------------------------------------------
@app.route("/performance")
def performance():
    return render_template(
        "performance.html",
        class_rows=performance_rows,
        overall=overall_metrics,
    )


if __name__ == "__main__":
    app.run(debug=True)
