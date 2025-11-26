import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import joblib

# 1. Load dataset
df = pd.read_csv("Language Detection.csv")

# Remove duplicates & nulls
df = df.drop_duplicates(subset=["Text"])
df = df.dropna()

# Convert text to lowercase
df["Text"] = df["Text"].astype(str).str.lower()

# 2. Vectorize using character-level TF-IDF
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4))
X = vectorizer.fit_transform(df["Text"])
y = df["Language"]

# 3. Balance Dataset using Oversampling
oversample = RandomOverSampler()
X_bal, y_bal = oversample.fit_resample(X, y)

print("Before balancing:", len(df))
print("After balancing:", len(y_bal))

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42
)

# 5. Logistic Regression (works best for this dataset)
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# 6. Save model + vectorizer
joblib.dump(model, "lang_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")
