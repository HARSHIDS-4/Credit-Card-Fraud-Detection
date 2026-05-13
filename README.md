# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using the Kaggle Credit Card Fraud Detection dataset.

## 📊 Dataset Source

**Dataset:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **Records:** 284,807 transactions
- **Features:** 30 (28 PCA-transformed features V1-V28 + Time + Amount)
- **Target:** Class (0 = Legitimate, 1 = Fraud)
- **Class Distribution:** Highly imbalanced (~0.172% fraudulent transactions)
- **Features:** All numeric features are normalized/PCA-transformed for privacy

> **Source:** The dataset was collected and analyzed by Worldline and the Machine Learning Group (ULB) of the Université Libre de Bruxelles as part of a research collaboration.

## 🎯 Business Problem

Credit card fraud detection is a critical challenge for financial institutions:
- **False Negatives (Missed Fraud):** Every missed fraud costs the bank real money
- **False Positives (Incorrect Flags):** Legitimate transactions flagged as fraud frustrate customers
- **Goal:** Maximize **Recall** (catch as many frauds as possible) while maintaining reasonable **Precision** (minimize false alarms)

## 📋 Project Structure

```
Credit-Card-Fraud-Detection/
├── Credit Card Fraud Detection.ipynb   # Main notebook with complete analysis
├── Credit_card_fraud_detection_model.pkl # Trained model (serialized)
├── README.md                           # Project documentation
└── LICENSE                             # MIT License
```

## 🔄 Project Approach

1. **Importing Libraries** - Set up all necessary Python packages
2. **Importing CSV File** - Load the Kaggle dataset
3. **Data Preprocessing** - Handle missing values, data types, and feature scaling
4. **Exploratory Data Analysis (EDA)** - Understand data distribution and class imbalance
5. **Feature Engineering & Train/Test Split** - Prepare data for modeling
6. **Model Pipeline** - Train and compare multiple models
7. **Save Best Model** - Persist the best performing model
8. **Conclusion** - Summary of findings and recommendations

## 💻 Technologies & Libraries

### Languages & Frameworks
- **Python 3.x**
- **Jupyter Notebook**

### Data Processing & Analysis
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing

### Machine Learning
- **scikit-learn** - Model training and evaluation
  - LogisticRegression
  - RandomForestClassifier
  - Feature selection (SelectKBest, f_classif)
  - RobustScaler for feature scaling
  - Pipeline for streamlined preprocessing

### Data Visualization
- **matplotlib** - Static plots and visualizations
- **seaborn** - Statistical data visualization

### Model Persistence
- **pickle** - Save and load trained models

## 📊 Model Performance Comparison

The following table shows the performance metrics of different classification models on the test set:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.9945 | 0.82 | 0.68 | 0.74 | 0.9680 |
| **Random Forest** | 0.9968 | 0.94 | 0.87 | 0.90 | 0.9892 |

### Metrics Explanation

- **Accuracy:** Overall correctness of predictions (less meaningful for imbalanced data)
- **Precision:** Of predicted fraud cases, how many were actually fraudulent
  - Important to minimize false positives (customer frustration)
- **Recall:** Of actual fraud cases, how many were detected
  - Critical metric - missing fraud costs money
- **F1-Score:** Harmonic mean of Precision and Recall
  - Balanced metric for imbalanced datasets
- **AUC-ROC:** Area under the Receiver Operating Characteristic curve
  - Measures model's ability to distinguish between classes

### Key Insights

- **Random Forest significantly outperforms Logistic Regression** across all metrics
- **High Recall (87%)** means the model catches most fraudulent transactions
- **High Precision (94%)** minimizes false alarms that frustrate customers
- The **highly imbalanced dataset** (0.172% fraud rate) requires careful evaluation beyond accuracy alone

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Notebook

1. Clone the repository
2. Navigate to the project directory
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `Credit Card Fraud Detection.ipynb`
5. Run the cells sequentially to reproduce the analysis

### Using the Trained Model

```python
import pickle

# Load the pre-trained model
with open('Credit_card_fraud_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions on new data
predictions = model.predict(X_new)
```

## 📈 Results & Findings

- Random Forest is the recommended model for production deployment
- The model achieves a good balance between fraud detection (recall) and customer satisfaction (precision)
- Feature scaling and PCA transformation significantly impact model performance
- Class imbalance handling through sampling techniques improves model robustness

## 🔍 Recommendations

1. **Deployment:** Use Random Forest model with threshold tuning for production
2. **Monitoring:** Implement continuous monitoring for model drift
3. **Data Imbalance:** Consider SMOTE or other oversampling techniques for further improvement
4. **Real-time:** Optimize model latency for real-time transaction processing
5. **Updates:** Retrain model periodically with new transaction data

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced Learning Techniques](https://imbalanced-learn.org/)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**HARSHIDS-4**

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues) if you want to contribute.

---

**Last Updated:** May 2026

For questions or feedback, please open an issue or submit a pull request.
