**Language Detection using NLP & Machine Learning**

A complete end-to-end Language Detection System built using TF-IDF (character n-grams), Random Oversampling, and Logistic Regression, designed to classify text into multiple languages with high accuracy.

This project includes:

✔ Dataset preprocessing
✔ Character-level TF-IDF vectorization
✔ Balanced training using RandomOverSampler
✔ Model training & evaluation
✔ A lightweight Python prediction script
✔ Interactive web UI / HTML interface
✔ Project report & presentation

Features

Multi-language text classification

Character-level TF-IDF: Works well even with short texts

Random Oversampling: Handles imbalanced dataset effectively

Logistic Regression: Fast, simple, and highly interpretable

Interactive UI (HTML/Python)

High accuracy & detailed classification metrics

Technologies Used
Category	Tools
Programming	Python
ML / NLP	Scikit-Learn, TF-IDF, Logistic Regression
Data Balancing	imbalanced-learn (RandomOverSampler)
Web UI	HTML, CSS
Others	Pandas, Joblib

📁 Project Structure (Suggested)
├── app.py                  
├── train_and_save_model.py   
├── Language Detection.csv    
├── model_compressed.pkl.gz   
├── vectorizer.pkl            
├── index.html                
├── performance.html          
├── docs/
│   ├── mini_project.pdf      
│   └── AML_report.pptx       
└── README.md                 
📊 Dataset

The dataset contains two columns:

Column	Description
Text	Input text
Language	Language label for that text

Example:

Text,Language
"Hello, how are you?",English
"Bonjour tout le monde",French
"नमस्ते दुनिया",Hindi


Duplicates and missing values are removed before training.

⚙️ How the Model Works
1. Preprocessing

Remove duplicates

Drop null rows

2. Vectorization

We use TF-IDF with character n-grams (2 to 4):

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4))


This captures patterns like
th, he, ell, bonjour, etc.

3. Balancing the classes

Imbalanced datasets can bias the model.
We use RandomOverSampler:

oversample = RandomOverSampler()
X_bal, y_bal = oversample.fit_resample(X_vec, y)

4. Training the model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

5. Evaluation

The model is evaluated using:

Accuracy

Precision

Recall

F1-Score

▶️ Running the Model
1. Install dependencies
pip install -r requirements.txt

2. Train & save the model
python train_and_save_model.py

3. Run the prediction app
python app.py

4. Use the HTML interface

Open:

index.html

🧪 Example Prediction
from app import predict_language

print(predict_language("Bonjour, comment allez-vous?"))


Output:

French

📈 Model Performance

A detailed classification report is available inside:

performance.html


Including precision, recall, and F1-score for each language.

📄 Documents

The project report and presentation slides are available in the docs/ section:

mini_project.pdf

AML_report.pptx

📝 Future Improvements

Deploy as a Streamlit Web App

Add a REST API using FastAPI

Use Deep Learning models (LSTM / BERT) for better accuracy

Support real-time detection in a chat interface

👨‍💻 Author

Athul S. Nair
B.Tech CSE – Jain University
Course: Advanced Machine Learning (23CSE514)
Mini Project: Language Detection using Machine Learning
