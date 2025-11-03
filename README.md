# Network Anomaly Detection (ML Project)

This project explores network anomaly detection using a small dataset and three classic machine learning models. The goal is to identify anomalous network activity based on features like latency and throughput.

This script loads the `network-logs.csv` dataset, trains three classifiers, and evaluates their performance using accuracy, F1 score, and sensitivity. It also visualizes the dataset to understand its characteristics.

## 🚀 Project Overview

The project follows these key steps:
1.  **Data Loading:** Loads the `network-logs.csv` dataset using pandas.
2.  **Data Preparation:** Separates the data into features (X) and a target (y: 'ANOMALY'), then splits it into 80% training and 20% testing sets.
3.  **Model Training:** Trains three different ML models on the data.
4.  **Model Evaluation:** Compares the models based on key performance metrics.
5.  **Data Visualization:** Creates plots to explore the data and the model results.

## 🤖 Models Implemented

This project compares three classifiers:
* **K-Nearest Neighbors (KNN)**
* **Gaussian Naive Bayes (GNB)**
* **Decision Tree**

## 📈 Evaluation Metrics

Models are evaluated using three key metrics:
* **Accuracy:** The percentage of correct predictions.
* **F1 Score:** The balanced average of precision and recall.
* **Sensitivity (Recall):** The model's ability to find *all* actual anomalies. This is a critical metric for security tasks.

## 📊 Visualizations

The script generates four plots to analyze the data and results:
1.  **Model Accuracy Comparison:** A bar chart showing the accuracy of each model.
2.  **Data Flow Scatter Plot:** Plots `LATENCY` vs. `THROUGHPUT` to visualize data clusters.
3.  **Outlier Detection Boxplot:** Shows the spread and outliers for `LATENCY` and `THROUGHPUT`.
4.  **Feature Distribution Histogram:** Shows the frequency of values for both features.

## 🔧 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git)
    cd YOUR-REPOSITORY-NAME
    ```

2.  **Install dependencies:**
    This project requires `pandas`, `scikit-learn`, and `matplotlib`. You can install them via `pip`:
    ```bash
    pip install pandas scikit-learn matplotlib
    ```

3.  **Get the dataset:**
    * You will need the `network-logs.csv` file.
    * Place the `.csv` file in the same directory as your Python script.

4.  **Update the dataset path:**
    If you placed the file in a different location, make sure to update this line in the script:
    ```python
    dataset_path = 'network-logs.csv' # <-- Change this path if needed
    ```

5.  **Run the script:**
    ```bash
    python Supervised_learning_for_network_log_dataset.py
    ```
