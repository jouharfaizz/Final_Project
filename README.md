# 🚗 Car Price Prediction using Machine Learning

Welcome to the **Car Price Prediction** project! This project demonstrates how machine learning can be used to estimate car prices based on features like brand, year, mileage, and engine size.

---

## 📁 Files

* `car_price_prediction.ipynb` – Main Jupyter notebook with full implementation
* `Used_Cars_Dataset.csv` – Dataset from [data.world](https://data.world/data-society/used-cars-data)
* `README.md` – Project overview

---

## 🧠 Objective

A Chinese automobile company plans to enter the U.S. market and wants to **predict optimal car prices** using historical data. This project includes:

* Data preprocessing and outlier removal
* Feature selection and scaling
* Model training and evaluation
* Hyperparameter tuning
* Saving the best model for future use

---

## 🔧 Technologies & Libraries

* Python
* pandas, numpy
* matplotlib, seaborn
* scikit-learn (linear\_model, tree, ensemble, svm, metrics, model\_selection)
* joblib

---

## 📊 Dataset Overview

* **Source**: [Used Car Dataset – data.world](https://data.world/data-society/used-cars-data)
* **Target**: `Price`
* **Key Features**:

  * Brand
  * Model
  * Year of registration
  * Mileage
  * Engine power
  * Fuel type
  * Transmission

---

## 🧼 Data Preprocessing

* Dropped irrelevant/missing columns
* Removed outliers using IQR method
* Selected top features using `SelectKBest` and `f_regression`
* Scaled numerical data using `StandardScaler`
* Split data into training and testing sets

---

## 🤖 Model Building

Algorithms used:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Regressor

---

## 📈 Model Evaluation

Evaluation Metrics:

* R² Score
* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

---

## 🔍 Hyperparameter Tuning

Used `GridSearchCV` to find the best parameters for models.

```python
from sklearn.model_selection import GridSearchCV
```

---

## 💾 Model Saving

The best-performing model is saved for deployment or future use.

```python
import joblib
joblib.dump(best_model, 'car_price_model.pkl')
```

---

## 📊 Visualizations

Included visualizations:

* Boxplots for outlier detection
* Heatmap for correlation analysis
* Distribution plots
* Actual vs. Predicted price scatter plots

---

## 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the notebook:

```bash
jupyter notebook
```

Open `car_price_prediction.ipynb` and run all cells.

---

## 🔮 Future Improvements

* Add deep learning models (TensorFlow, Keras)
* Location-aware pricing
* Deploy the model as a web application using Flask or Streamlit

---

Let me know if you'd like this saved as a `.md` file or customized further.
