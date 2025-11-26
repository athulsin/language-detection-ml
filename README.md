Language Detection using NLP & Machine Learning

A complete end-to-end Language Detection System built using TF-IDF (character n-grams), Random Oversampling, and Logistic Regression, designed to classify text into multiple languages with high accuracy.

This project includes:

✔ Dataset preprocessing
✔ Character-level TF-IDF vectorization
✔ Balanced training using RandomOverSampler
✔ Model training & evaluation
✔ A lightweight Python prediction script
✔ Interactive web UI (HTML)
✔ Project report & presentation
✔ Screenshots & sample outputs

📸 Screenshots & Demo
🔹 1. GitHub Project Structure

Upload this screenshot as: screenshots/repo_structure.png

![Project Structure](screenshots/repo_structure.png)

🔹 2. Sample Prediction Output (Python)

Upload as: screenshots/python_prediction.png

![Python Prediction Output](screenshots/python_prediction.png)

🔹 3. HTML Interface Screenshot

Upload as: screenshots/html_ui.png

![HTML UI Screenshot](screenshots/html_ui.png)

🔹 4. Model Performance Metrics

Upload as: screenshots/performance_report.png

![Performance Report](screenshots/performance_report.png)

📁 Project Structure
├── app.py                    # Prediction / Web interface script
├── train_and_save_model.py   # Model training script
├── Language Detection.csv    # Dataset
├── model_compressed.pkl.gz   # Trained model
├── vectorizer.pkl            # Saved TF-IDF vectorizer
├── index.html                # Web UI for text input
├── performance.html          # Model evaluation visual report
├── docs/
│   ├── mini_project.pdf      # Project report
│   └── AML_report.pptx       # Presentation PPT
├── screenshots/
│   ├── repo_structure.png
│   ├── python_prediction.png
│   ├── html_ui.png
│   └── performance_report.png
└── README.md                 # Project documentation

🧠 Technologies Used
Category	Tools
Programming	Python
ML / NLP	Scikit-Learn, TF-IDF, Logistic Regression
Data Balancing	imbalanced-learn (RandomOverSampler)
Web UI	HTML, CSS
Evaluation	Classification Report, Accuracy Score
⚙️ How the Model Works
1. Preprocessing

Remove duplicate rows

Remove null rows

2. TF-IDF Vectorization

Using character n-grams (2 to 4):

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4))

3. Balancing

Using:

RandomOverSampler()

4. Logistic Regression Model

Trained with:

LogisticRegression(max_iter=2000)

5. Evaluation

Metrics available in performance.html.

▶️ Running the Project
Install dependencies
pip install -r requirements.txt

Train & save model
python train_and_save_model.py

Run prediction
python app.py

Open HTML UI

Open:

index.html

🧪 Example Usage
from app import predict_language
print(predict_language("Bonjour tout le monde"))


Output:

French

📊 Model Performance

A detailed performance result is available in:

performance.html

Screenshot included in the screenshots folder

📄 Documents

Available in /docs:

mini_project.pdf

AML_report.pptx

📝 Future Improvements

Deploy with Streamlit

Add FastAPI REST endpoint

Use Deep Learning models (LSTM, BERT)

Create a mobile app version

👨‍💻 Author

Athul S. Nair
B.Tech CSE – Jain University
Course: Advanced Machine Learning (23CSE514)
Mini Project: Language Detection using Machine Learning
