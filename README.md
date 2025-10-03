# Leukemia Subtype Prediction using LASSO and Multi-Layer Perceptron (MLP)

This project focuses on predicting leukemia subtypes from high-dimensional gene expression data. It employs **LASSO (Least Absolute Shrinkage and Selection Operator)** for crucial feature selection to manage the large number of genes, followed by a **Multi-Layer Perceptron (MLP)** classifier for prediction.

The primary goal is to demonstrate that dimensionality reduction via LASSO can significantly improve model performance and simplify the model structure, particularly when evaluated using **Balanced Accuracy**, a metric suitable for the dataset's imbalanced classes.

---

## Dataset

* **Dataset Name:** Leukemia gene expression - CuMiDa
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/brunogrisci/leukemia-gene-expression-cumida)
* **Type:** High-dimensional data (Gene Expression Microarray)
* **Dimensions:** 64 samples (rows) across 22,285 features (columns, including auxiliary columns). The gene expression data itself has **22,283 gene probes**.

### Target Variable (Leukemia Subtype)
The original categorical 'type' column was encoded into numerical categories:

| Original Subtype | Encoded Value | Sample Count |
| :--------------- | :------------ | :----------- |
| `AML`            | 3             | 26           |
| `Bone_Marrow`    | 2             | 10           |
| `PB`             | 4             | 10           |
| `PBSC_CD34`      | 5             | 10           |
| `Bone_Marrow_CD34` | 1             | 8            |

---

## Project Goals

1.  **Feature Selection:** Use **LASSO (L1 regularization)** within a Multinomial Logistic Regression model to identify a subset of the most relevant genes (features) for leukemia subtype classification.
2.  **Model Building:** Train a **Multi-Layer Perceptron (MLP)** classifier.
    * One model (`mlp_all`) will use all 22,283 scaled gene features.
    * A second model (`mlp_21`) will use only the genes selected by LASSO.
3.  **Evaluation:** Evaluate and compare both MLP models using the **Balanced Accuracy** metric, as it is appropriate for the imbalanced nature of the target classes.

---

## ⚙️ Methodology and Key Steps

### 1. Data Loading and Preprocessing
* The raw data was loaded from `Leukemia_GSE9476.csv`.
* The categorical `type` column was converted to numerical categories (1 to 5) and set as the target variable ($\mathbf{y}$).
* The irrelevant `samples` column was dropped.
* The input features ($\mathbf{X}$) were separated from the target variable ($\mathbf{y}$).
* **Scaling:** All gene expression data were scaled using **`MinMaxScaler`**.
* **Splitting:** The data was split into training and testing sets with a **60/40 ratio (`test_size=0.4`)** and `random_state=1234`.

### 2. Feature Selection with LASSO
* A **Multinomial Logistic Regression** model with **L1 penalty** (`penalty='l1'`, `solver='saga'`, `C=0.2`, `max_iter=200`) was fitted to the training data.
* The LASSO regularization forces the coefficients of less relevant features to zero, thereby performing feature selection.
* **Result:** LASSO successfully reduced the feature space from **22,283 columns to 21 selected gene columns**.

### 3. Machine Learning Model (MLP)
* A **Multi-Layer Perceptron (`MLPClassifier`)** was used for classification.
* **Configuration:** `hidden_unit=64`, `max_iter=700`, `random_state=42`.

---

## Results and Conclusions

### Selected Gene Features (21 Genes)
The following 21 gene probe columns were selected by the LASSO model:

* `200736_s_at`
* `201105_at`
* `201765_s_at`
* `201850_at`
* `204007_at`
* `205500_at`
* `206522_at`
* `206674_at`
* `207008_at`
* `209062_x_at`
* `209395_at`
* `210119_at`
* `210376_x_at`
* `211163_s_at`
* `212052_s_at`
* `219672_at`
* `221345_at`
* `221754_s_at`
* `44040_at`
* `AFFX-HUMGAPDH/M33197_5_at`
* `AFFX-HUMGAPDH/M33197_M_at`

### Model Performance (Balanced Accuracy)

| Model Name | Feature Count | Training Balanced Accuracy | Testing Balanced Accuracy |
| :--- | :--- | :--- | :--- |
| **Single MLP** | 22,283 | 0.4125 | 0.4000 |
| **LASSO + MLP** | 21 | **1.0000** | **0.9600** |

The performance comparison clearly shows the benefit of the feature selection:

### Key Takeaways

1.  **Dimensionality Reduction:** LASSO was highly effective in reducing the complexity of the dataset by selecting a minimal set of 21 highly predictive genes out of 22,283.
2.  **Performance Increase:** The MLP model trained on the 21 selected genes achieved dramatically higher Balanced Accuracy on the test set ($\mathbf{0.96}$ vs $\mathbf{0.40}$).
3.  **Model Simplicity:** Using the 21 selected gene columns by LASSO results in a more interpretable and computationally efficient MLP model.
4.  **Overfitting Potential:** The training accuracy of $1.0000$ for the LASSO+MLP model suggests potential **overfitting** to the training data, despite the good test performance. Future development should focus on techniques like **cross-validation, hyperparameter tuning,** or **early stopping** to ensure robust generalization.

---

## Requirements

The project notebook was run using Python and requires the following libraries:

```bash
pandas
numpy
sklearn
matplotlib
kagglehub # Used for initial data download, but data loaded from csv in the notebook
