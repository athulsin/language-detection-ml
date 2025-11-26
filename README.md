**Language Detection using NLP & Machine Learning**

A complete end-to-end Language Detection System built using Machine Learning and Natural Language Processing techniques.
The model uses TF-IDF (character n-grams) + Logistic Regression, with Random Oversampling to handle class imbalance.

📸 Screenshots
🔹 Web Interface (Home Page)

🔹 Model Performance & Evaluation

📁 Project Structure
├── src/
│   ├── app.py                    # Prediction / Web interface script
│   └── train_and_save_model.py   # Model training script
├── data/
│   └── Language Detection.csv    # Dataset
├── models/
│   ├── vectorizer.pkl            # TF-IDF vectorizer
│   └── model_compressed.pkl.gz   # Trained model
├── html/
│   ├── index.html                # Web UI
│   └── performance.html          # Evaluation output
├── docs/
│   ├── mini_project.pdf          # Project report
│   └── AML_report.pptx           # Presentation
├── screenshots/
│   ├── home.png                  # UI screenshot
│   └── model.png                 # Model evaluation screenshot
└── README.md

🧠 Overview

This project detects the language of any input text using:

Character-level TF-IDF

n-gram range = (2,4)

Logistic Regression classifier

RandomOverSampler for class balancing

The model performs well even on short text, because character patterns like th, na, ell, que, नम help identify languages accurately.

⚙️ How the Model Works
1️⃣ Data Cleaning

Remove duplicate text

Remove missing values

2️⃣ Feature Extraction – TF-IDF
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4))

3️⃣ Balancing the Dataset
oversample = RandomOverSampler()
X_bal, y_bal = oversample.fit_resample(X_vec, y)

4️⃣ Model Training
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

5️⃣ Evaluation

Accuracy, precision, recall, and F1-score are generated.
You can view them in:

html/performance.html

▶️ Running the Project
Install dependencies
pip install -r requirements.txt

Train the model
python src/train_and_save_model.py

Run the prediction app
python src/app.py

Use the HTML UI

Open:

html/index.html

🧪 Example Usage
from src.app import predict_language
print(predict_language("Bonjour tout le monde"))


Output:

French

📄 Documents

All documentation files are in the docs/ folder:

mini_project.pdf

AML_report.pptx

📝 Future Improvements

Upgrade to Deep Learning models (BERT / LSTM)

Deploy using Streamlit / FastAPI

Build a mobile app version

👨‍💻 Author

Athul S. Nair
B.Tech CSE – Jain University
Course: Advanced Machine Learning (23CSE514)
Mini Project: Language Detection using Machine Learning
