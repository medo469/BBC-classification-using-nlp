<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# BBC News Text Classification

A complete project to classify BBC News articles using classical and deep learning models.
The notebook contains data loading, preprocessing, baseline modeling using TF-IDF + Naive Bayes, advanced classification using DistilBERT, and a simple Streamlit app to test the model interactively.

***

## 📂 Project Structure

```
project/
│
├── assignment.ipynb      # Full Jupyter notebook with data processing, modeling, evaluation
├── app.py                # Streamlit app for news article classification
├── data/                 # Folder to place the BBC news dataset file (bbc-news-data.csv)
├── README.md             # This file
```


***

## 🚀 How to Run the Project

### 1. Download Dataset

- Download the BBC News dataset at:
[UCD ML Group BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html)
or
[Kaggle BBC News Dataset](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive)
- Save the dataset CSV file (e.g., `bbc-news-data.csv`) inside the project's `data/` folder.


### 2. Install Required Packages

Run the following command in your terminal or Anaconda prompt to install dependencies:

```bash
pip install pandas scikit-learn nltk spacy torch transformers streamlit
python -m spacy download en_core_web_sm
```


***

### 3. Run Jupyter Notebook (`assignment.ipynb`)

- Open the notebook in Jupyter, VS Code, JupyterLab, or Google Colab.
- Run cells sequentially:
    - Data loading and merging title + content
    - Preprocessing pipeline:
        - Lowercasing
        - Removing punctuation and stopwords
        - Stemming (PorterStemmer) and Lemmatization (spaCy)
    - Feature Extraction using TF-IDF
    - Splitting dataset into training/testing
    - Training a baseline Naive Bayes classifier
    - Model evaluation with accuracy, precision, recall, and confusion matrix

***

### 4. Streamlit App (`app.py`)

A simple web app for interactive classification using the trained DistilBERT model.

**How to run:**

```bash
streamlit run app.py
```

- Paste or type a news article into the text area.
- Click **Classify**.
- The app shows the predicted category and confidence.

**Note:**
Make sure you have the trained model, tokenizer, and label encoder loaded in the same environment or modify the app to load saved model files.

***

## ⚙️ Explanation

- **assignment.ipynb:**
Contains all code to clean, preprocess, and model using traditional ML (TF-IDF + Naive Bayes). Includes improvements by fine-tuning a DistilBERT transformer implemented with PyTorch \& HuggingFace Transformers library.
- **app.py:**
A minimal Streamlit interface connected to the DistilBERT classifier for live text prediction.

***

## ⚖️ Ethics \& Limitations

- The dataset is limited to BBC News articles, which might introduce bias favoring topics common in that source.
- Models can misclassify articles, potentially misleading users if used blindly.
- Mitigate risks by:
    - Using diverse datasets.
    - Regularly monitoring and updating the model.
    - Informing end users about limitations and uncertainties.

***

## 📝 Troubleshooting

- If `spacy` throws errors, make sure to run the download command for the English model:

```bash
python -m spacy download en_core_web_sm
```

- If the dataset file is not found, check the path in the notebook matches the local location of your downloaded data.
- For Streamlit app errors, verify your Python environment has all dependencies installed.

***

## 👩💻 Contributions and Questions

If you find issues or want to improve the project, please open an issue or submit a pull request.
For help or questions, reach out in the project's GitHub discussions or contact me directly!

***

**Author:** [Your Name or GitHub handle]
_Sudanese ML Enthusiast_

***

لو عايز ترجمة أو شرح إضافي لأي جزء في الـREADME أو محتاج زي ما هو بالعربي، قول لي!

<div style="text-align: center">⁂</div>

[^1]: assignment.ipynb

[^2]: app.py

