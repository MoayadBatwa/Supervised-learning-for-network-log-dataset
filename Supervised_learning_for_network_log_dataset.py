# -----------------------------------------------------------------
# Step 1: Import necessary libraries
# -----------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Step 2: Loading data from CSV file
# -----------------------------------------------------------------
# !!! IMPORTANT: Replace this path with the actual path to your dataset !!!
dataset_path = 'network-logs.csv'
df = pd.read_csv(dataset_path)

# Displaying a sample of the data
print("Sample of the dataset:")
print(df.head())

# -----------------------------------------------------------------
# Step 3: Separating features (X) and target (y)
# -----------------------------------------------------------------
X = df.drop(columns=['ANOMALY'])
y = df['ANOMALY']

# -----------------------------------------------------------------
# Step 4: Splitting the dataset (80% train, 20% test)
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------------
# Step 5: Initializing models
# -----------------------------------------------------------------
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# -----------------------------------------------------------------
# Step 6: Training and evaluating models
# -----------------------------------------------------------------
results = {} # Dictionary to store accuracy for the bar plot

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted') # Use 'weighted' for general case
    sensitivity = recall_score(y_test, y_pred, average='weighted') # Use 'weighted' for general case
    
    # Store accuracy for plotting
    results[model_name] = accuracy
    
    # Print performance
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------
# Step 7: Visualization of results
# -----------------------------------------------------------------

# --- Bar Chart: Model Accuracy Comparison ---
plt.figure(figsize=(5, 3))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# --- Scatter Plot: LATENCY vs THROUGHPUT ---
plt.figure(figsize=(4, 3))
# Note: This plots from the original 'X' (all data), not just train or test
plt.scatter(X['LATENCY'], X['THROUGHPUT'], alpha=0.6)
plt.xlabel('LATENCY')
plt.ylabel('THROUGHPUT')
plt.title('DATA FLOW (LATENCY vs THROUGHPUT)')
plt.show()

# --- Boxplot: Detect Outliers ---
plt.figure(figsize=(4, 3))
plt.boxplot([X['LATENCY'], X['THROUGHPUT']], labels=['LATENCY', 'THROUGHPUT'])
plt.title('Outliers Detection')
plt.show()

# --- Histogram: Feature Distributions ---
plt.figure(figsize=(4, 3))
plt.hist(X['LATENCY'], bins=20, alpha=0.7, label='LATENCY')
plt.hist(X['THROUGHPUT'], bins=20, alpha=0.7, label='THROUGHPUT')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of LATENCY and THROUGHPUT')
plt.legend()
plt.show()

# -----------------------------------------------------------------
# Final Line: End Message
# -----------------------------------------------------------------
print("\nExperiment complete!")