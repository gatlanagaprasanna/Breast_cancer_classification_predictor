# Breast_cancer_classification_predictor
 
# Breast Cancer Classification Predictor

A simple and user-friendly web app built with **Streamlit** to predict whether a breast tumor is **Malignant** or **Benign** based on input medical features. The model is trained on the Breast Cancer Wisconsin dataset using machine learning techniques.

---

## Features

- Clean web interface built with **Streamlit**
- Uses a trained **scikit-learn** model
- Real-time predictions
- Easy to use for demonstration and educational purposes

---

## Technologies Used

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- pickle (for saving the model)

---

## Dataset

- **Source:** [Breast Cancer Wisconsin (Diagnostic) Data Set](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Features:** 30 numerical features describing characteristics of the cell nuclei in breast mass
- **Target:** Diagnosis (`0` = malignant, `1` = benign)

---

## Features Used

Here are a few of the 30 features used in prediction:

- Mean radius  
- Mean texture  
- Mean perimeter  
- Mean area  
- Mean smoothness  
*(and more...)*

---

## How to Run Locally

###  Clone the Repository

```bash
git clone https://github.com/your-username/breast-cancer-predictor.git
cd breast-cancer-predictor/App


## 🛠️ How to Install Dependencies

Make sure you have **Python 3.8+** installed. Then install all required Python packages using:

```bash
pip install -r requirements.txt

## Run the web app

streamlit run app.py

