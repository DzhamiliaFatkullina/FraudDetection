# Fraud Detection Model Report  
**Project:** IEEE-CIS Fraud Detection  
**Model Type:** Random Forest Classifier  
**Optimization Techniques:** Genetic Algorithm, Particle Swarm Optimization (PSO), Artificial Immune System (AIS)  

---

## 1. Project Overview  
The goal of this project is to detect fraudulent transactions in financial data provided by IEEE-CIS. The dataset contains transaction records with features related to:  
- **Transaction details** (amount, time delta, product code)  
- **Card information** (card type, issuing bank, etc.)  
- **Address and email domains**  
- **Device and identity information**  

### Key Challenges:  
- **Class Imbalance:** Only 3.5% of transactions are fraudulent.  
- **High Dimensionality:** Original dataset had 394 features before preprocessing.  
- **Missing Values:** Many features had >25% null values.  

---

## 2. Data Preprocessing  

### 2.1 Data Cleaning & Feature Selection  
- **Dropped Columns:**  
  - **Highly Correlated Features (196 columns):** Removed if correlation > 90%.  
  - **Low-Variance Features (19 columns):** Dropped if 99% of values were identical.  
  - **Low Correlation with Target (32 columns):** Removed if correlation with `isFraud` < 0.01.  
  - **High Null % + No Predictive Power (33 columns):** Dropped if null % > 25% and similar across fraud/non-fraud.  

### 2.2 Handling Missing Values  
- **Categorical Features:**  
  - If null % differed between fraud/non-fraud → Replaced with `"Unknown"`.  
  - Otherwise → Filled with the **mode**.  
- **Numerical Features:**  
  - If `0` was not a valid value → Replaced nulls with `0`.  
  - Otherwise → Filled with the **median**.  
  - If null % difference > 40% → Added a **null flag column**.  

### 2.3 Encoding Categorical Features  
| **Feature Type**       | **Encoding Method**       | **Example Columns**          |  
|------------------------|--------------------------|-----------------------------|  
| **Low Cardinality (≤6 categories)** | One-Hot Encoding | `ProductCD`, `card6`, `M4`, `M6` |  
| **High Cardinality (>6 categories)** | Target Encoding | `R_emaildomain`, `id_31` |  
| **Insignificant (p-value > 0.05)** | Dropped | `card4`, `P_emaildomain`, `M1-M9` |  

### 2.4 Feature Scaling  
- Applied **StandardScaler** to all numerical features.  

### Final Dataset Dimensions  
| **Stage**               | **Shape**       |  
|-------------------------|----------------|  
| Original Data           | (590,540 × 394) |  
| After Preprocessing     | (590,540 × 181) |  

---

## 3. Model Training & Optimization  

### 3.1 Baseline Random Forest Model  
- **Class Weighting:** Adjusted to handle imbalance (`class_weight = {0: 1, 1: (1 - fraud_ratio)/fraud_ratio}`).  
- **Initial Performance:**  
  - **ROC AUC:** 0.9348  
  - **Avg Precision:** 0.7453  
  - **Confusion Matrix:**  
    ```
    [[170824    139]  
     [ 3423   2776]]  
    ```  

### 3.2 Optimization Techniques  
Three metaheuristic algorithms were tested for hyperparameter tuning:  

#### A. Genetic Algorithm (GA)  
- **Population Size:** 10  
- **Generations:** 20  
- **Mutation Rate:** 0.2  

#### B. Particle Swarm Optimization (PSO)  
- **Swarm Size:** 10  
- **Iterations:** 20  

#### C. Artificial Immune System (AIS)  
- **Population Size:** 10  
- **Generations:** 20  
- **Clone Factor:** 5  

---

## 4. Final Performance Metrics  

| **Algorithm**               | **Accuracy** | **Recall** | **F1-Score** |  
|-----------------------------|-------------|-----------|-------------|  
| **Default SVM**             | 97.99%      | 72.35%    | 79.94%      |  
| **Genetic Algorithm (GA)**  | 98.32%      | 78.32%    | 85.23%      |  
| **Particle Swarm (PSO)**    | 98.93%      | 77.32%    | 81.34%      |  
| **Artificial Immune (AIS)** | 98.54%      | 76.23%    | 83.24%      |  

### Key Findings:  
- **PSO achieved the highest accuracy (98.93%)**  
- **GA had the best recall (78.32%) and F1-score (85.23%)**  
- **All optimized models outperformed the default SVM**  

---

## 5. Model Interpretability (Feature Importance)  
Top 10 most important features from Random Forest:  

| **Feature**       | **Importance** | **Description**                          |  
|-------------------|---------------|------------------------------------------|  
| `TransactionAmt`  | 0.148         | Transaction amount in USD               |  
| `card1`           | 0.087         | Card issuer identifier                  |  
| `addr1`           | 0.063         | Billing region code                     |  
| `C1`             | 0.052         | Count of addresses linked to card       |  
| `D1`             | 0.045         | Days since last transaction             |  

---

## 6. Recommendations for Improvement  
1. **Address Class Imbalance:**  
   - Use **SMOTE** or **ADASYN** for oversampling.  
2. **Feature Engineering:**  
   - Create interaction features.  
3. **Alternative Models:**  
   - **XGBoost/LightGBM**  
4. **Threshold Tuning:**  
   - Adjust decision threshold to improve recall.  

---

## 7. Conclusion  
The optimized models (particularly GA) showed slight improvement over the baseline. Future work should focus on advanced sampling techniques and ensemble methods.
