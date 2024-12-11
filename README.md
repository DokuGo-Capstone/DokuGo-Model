# **DokuGO Machine Learning**

**DokuGO Machine Learning** is a deep learning-powered solution designed to assist users in managing personal finances by providing predictions for monthly and weekly expenses. This repository contains the backend, deep learning models, dataset, and notebook.

---

## **Features**

- **Financial Prediction**: Implements a Neural Network to predict personal financial patterns, including expenses and savings, based on user input.
- **Feedback Loop**: Utilizes **Retrieval-Augmented Generation (RAG)** to enhance prediction accuracy and context-aware financial insights through retraining with updated user data.

---

## **Installation**

To set up the project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/DokuGo-Capstone/DokuGo-Model.git
cd DokuGo-Model
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Linux
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## **Folder Structure**

```plaintext
DokuGo-Model/
├── Data/
│   ├── expenses_income_summary.csv
├── bayesian_tuning/
|   |   ├── expense_prediction
|   |   |    ├── trial_00 - 19
├── app.py
├── README.md
├── expense_prediction_model_fixed.h5
├── main.ipynb
└── requirements.txt
```

---

## **Dependencies**

All required dependencies are listed in `requirements.txt`. Key dependencies include:

- **TensorFlow**: For building and training machine learning models.
- **Flask**: For creating the API to deploy the models.
- **Pandas**: For data manipulation and analysis.
- **scikit-learn**: For preprocessing and evaluation metrics.

---

