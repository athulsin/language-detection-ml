**Language Detection using NLP & Machine Learning
**
A complete end-to-end Language Detection System built using TF-IDF, Logistic Regression, and Random Oversampling.

📸 Screenshots
Home Page

Model Performance

📁 Project Structure
├── src/
│   ├── app.py
│   └── train_and_save_model.py
├── data/
│   └── Language Detection.csv
├── models/
│   ├── vectorizer.pkl
│   └── model_compressed.pkl.gz
├── html/
│   ├── index.html
│   └── performance.html
├── docs/
│   ├── mini_project.pdf
│   └── AML_report.pptx
└── screenshots/
    ├── home.png
    └── model.png

🧠 Overview

This project detects the language of input text using:

Character-level TF-IDF

N-grams (2 to 4)

Logistic Regression

Random Oversampling for class balance

⚙️ How It Works
1. Data Cleaning

Remove duplicates

Remove missing rows

2. TF-IDF Vectorization
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4))

3. Balancing the Dataset
oversample = RandomOverSampler()
X_bal, y_bal = oversample.fit_resample(X_vec, y)

4. Training
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

▶️ Running the Project
Install Dependencies
pip install -r requirements.txt

Train the Model
python src/train_and_save_model.py

Run the Prediction App
python src/app.py

Open HTML UI

Open this file in your browser:

html/index.html

🧪 Example Usage
from src.app import predict_language
print(predict_language("Bonjour tout le monde"))


Output:

French

📄 Documents

Located in docs/:

mini_project.pdf

AML_report.pptx

📝 Future Enhancements

Streamlit deployment

FastAPI REST API

BERT-based language detection

Mobile app wrapper

👤 Author

Athul S. Nair
Mini Project – Advanced Machine Learning (23CSE514)
B.Tech CSE – Jain University
