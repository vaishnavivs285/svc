# 📧 Spam vs Non-Spam Email Detection using Non-Linear SVM

This project implements a **Spam vs Non-Spam (Ham) text classification system** using a **Support Vector Machine (SVM)** with a **non-linear RBF kernel**.  
The project demonstrates why **linear models fail on text data** and how **kernel SVMs** can handle non-linear decision boundaries effectively.

---

## 🔍 Problem Statement

Spam detection is a classic **text classification** problem where messages cannot be separated using a simple linear boundary.  
Word interactions such as *“free” + “win”* create non-linear patterns, making this an ideal use case for **kernel-based SVMs**.

---

## 📂 Dataset

- **Name:** SMS Spam Collection  
- **Source:** UCI Machine Learning Repository  
- **Total Samples:** ~5,572 SMS messages  
- **Classes:**
  - `ham` → Non-Spam (0)
  - `spam` → Spam (1)

The dataset is naturally **imbalanced**, with more ham messages than spam.

---

## 🗂 Project Structure

```
SVMSPAMMAIL/
│
├── data/
│   └── spam.csv
│
├── notebook/
│   └── svm.ipynb
│
├── models/
│   ├── spam_model.joblib
│   └── tfidf_vectorizer.joblib
│
└── README.md
```

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Jupyter Notebook (VS Code)

---

## 🧠 Approach

1. **Data Loading**
   - Loaded dataset using relative paths for portability.

2. **Text Vectorization**
   - Converted raw SMS text into numerical features using **TF-IDF Vectorizer**.
   - Removed English stopwords.
   - Limited feature size to avoid overfitting.

3. **Baseline Model**
   - Trained a **Linear SVM** to show its limitations on text data.

4. **Non-Linear Model**
   - Trained an **RBF Kernel SVM** to capture complex decision boundaries.

5. **Hyperparameter Tuning**
   - Tuned `C` and `gamma` using **GridSearchCV**.
   - Used **F1-score** due to class imbalance.

6. **Evaluation**
   - Evaluated model using Precision, Recall, and F1-score.

7. **Model Saving**
   - Saved trained model and TF-IDF vectorizer using `joblib`.

---

## 📈 Key Concepts Demonstrated

- Non-linear nature of text data
- Linear vs Kernel SVM comparison
- Hyperparameter tuning (`C`, `gamma`)
- Handling imbalanced datasets


---

## 🧪 How to Run

1. Clone the repository
2. Open the project folder in VS Code
3. Open:
   ```
   notebook/svm.ipynb
   ```
4. Run all cells in order

### Install dependencies
```
pip install pandas numpy scikit-learn matplotlib
```

---

## 🚀 Future Improvements

- Polynomial kernel comparison
- Class-weight handling for imbalance
- Deployment using Streamlit or Flask
- Testing on private real email data

---


