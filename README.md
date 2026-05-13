# DataScience_Complete_Project

This is a comprehensive Data Science project that follows the below process:

1. Establish a goal.
2. Understand the data
3. Conduct data exploration.
4. Data cleaning and transformation
5. Building of models
6. Obtaining and analyzing metrics
7. Comparing Models


# 🚲 Bike Rental Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![Pandas](https://img.shields.io/badge/Library-Pandas-orange) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow) ![TensorFlow](https://img.shields.io/badge/Library-TensorFlow-ff6f00) ![Plotly](https://img.shields.io/badge/Visualization-Plotly-232F3E)

## 📌 Project Objective
The goal of this project is to **predict the number of bike rentals** by exploring which environmental, temporal, and social variables most significantly affect rental demand. This study follows a full Data Science pipeline, from raw data ingestion to advanced predictive modeling.

## 🚀 Phase 1: Data Engineering & EDA

### 1. Data Understanding & Cleaning
*   **Feature Renaming:** Technical column names (e.g., `atemp`, `hum`, `cnt`) were transformed into descriptive labels to ensure code maintainability and clarity.
*   **Data Decoding:** Normalized values for temperature, wind speed, and humidity were reverted to their original scales ($°C$, $km/h$, and $\%$), allowing for a physically meaningful interpretation of the data.
*   **Integrity Check:** Performed rigorous checks for missing values and duplicates to ensure a clean dataset for statistical analysis.

### 2. Exploratory Data Analysis (EDA)
Using `Matplotlib`, `Seaborn`, and `Plotly`, I extracted key behavioral insights:
*   **Rider Segmentation:** Analysis of **Casual vs. Registered** riders, showing a strong dominance of registered users (over 80%) across both 2011 and 2012.
*   **Temporal Dynamics:**
    *   **Hourly Patterns:** Identified peak demand during commuting hours (8 AM and 5 PM) on workdays, contrasting with a steady midday peak on weekends.
    *   **Seasonality:** Analyzed how weather conditions and seasons (Winter, Spring, Summer, Autumn) shift the rental baseline.
*   **Environmental Impact:** Explored the correlation between rental counts and weather metrics like humidity and wind speed using area and bar charts.

### 3. Feature Selection & Statistical Refinement
To prepare the data for Machine Learning, I applied several advanced techniques:
*   **Outlier Management:** Applied **Winsorization** to handle extreme values in wind speed and humidity, preserving data volume while reducing noise.
*   **Multicollinearity (VIF):** Calculated the **Variance Inflation Factor** to detect redundant features. Variables like `adjusted_temperature` and `season` were removed to prevent model instability.
*   **Sequential Feature Selection (SFS):** Implemented a *Backward Elimination* strategy to isolate the top 10 features with the highest predictive power for the regression models.

---

## 🛠️ Tech Stack
*   **Data Manipulation:** `Pandas`, `NumPy`
*   **Statistical Analysis:** `Statsmodels`, `SciPy`
*   **Visualization:** `Plotly Express`, `Seaborn`, `Yellowbrick`
*   **Machine Learning (Pre-processing):** `Scikit-learn` (StandardScaler, PolynomialFeatures, SFS)

---

This is the second and final part of your **README.md** for the Bike Rental Project. This section focuses on the advanced modeling phase, including statistical transformations, multiple machine learning algorithms, hyperparameter tuning, and deep learning.

You can append this directly to the previous English version I created for you.

---

## 🤖 Phase 2: Advanced Modeling & Evaluation

### 1. Statistical Analysis & Target Transformation

Before modeling, the distribution of the target variable (`count`) was analyzed.

* **Normality Tests:** Applied **Jarque-Bera** and **Augmented Dickey-Fuller** tests to check for stationarity and distribution shapes.
* **Variance Stabilization:** To correct skewness, I experimented with three transformations:
* **Logarithmic Scale**
* **Square Root**
* **Box-Cox Transformation** (Selected for significantly reducing skewness and improving model performance).



### 2. Machine Learning Models

I implemented and compared several regression architectures to find the best fit for the data:

#### **A. Linear & Polynomial Regression**

* **Multiple Linear Regression:** Used as a baseline model after target transformation.
* **Backward Elimination:** Refined the model by removing non-significant features ($p > 0.05$).
* **Polynomial Regression:** Applied a 3rd-degree polynomial expansion to capture non-linear relationships, followed by **Standard Scaling** for numerical stability.

#### **B. Tree-Based Models**

* **Decision Tree Regressor:** Explored basic non-linear splits and analyzed **Feature Importance** to identify key drivers of demand.
* **Random Forest Regressor:** Built an ensemble of 100 trees to reduce variance and improve accuracy.
* **Validation:** Used **K-Fold Cross-Validation** ($k=5$ and $k=10$) to ensure the models generalize well to unseen data.

#### **C. Deep Learning (ANN)**

* **Architecture:** Developed a **Sequential Neural Network** using **TensorFlow/Keras**.
* **Layers:** Three hidden dense layers with **ReLU** activation.
* **Custom Metrics:** Implemented a custom $R^2$ score function to monitor the training process.
* **Training Analysis:** Visualized the **Loss Function (MSE)** and $R^2$ convergence across 500 epochs to detect overfitting vs. validation loss.

### 3. Hyperparameter Tuning

To maximize performance, the models were optimized using:

* **Randomized Search CV:** Efficiently scanned the parameter space for `max_depth`, `min_samples_split`, and `n_estimators`.
* **Grid Search CV:** Performed an exhaustive search around the best parameters found by Randomized Search to fine-tune the Random Forest model.
* **Results:** Achieved significant percentage improvements in **RMSE** and **MAE** compared to default configurations.

### 4. Model Diagnostics

* **Residual Analysis:** Used **Residual Plots** and the **Breusch-Pagan Test** to check for Heteroscedasticity.
* **Actual vs. Predicted Visualization:** Implemented interactive **Plotly scatter plots** to compare the test values against model predictions, highlighting the high accuracy achieved ($R^2$ scores).

---

## 📊 Summary of Results

The project demonstrates that while linear models provide a solid baseline, **Ensemble Methods (Random Forest)** and **Deep Learning** architectures provide the most robust predictions by capturing the complex, non-linear interactions between weather patterns and human behavior.

---

**This project showcases a complete Data Science workflow: from raw data cleaning and statistical inference to advanced machine learning optimization and neural network implementation.**
