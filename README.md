import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate traffic data
def generate_synthetic_data():
    np.random.seed(42)
    # Generating random traffic data (flow duration, byte count, packet count, etc.)
    n_samples = 1000
    flow_duration = np.random.uniform(0.1, 100, n_samples)
    byte_count = np.random.uniform(100, 5000, n_samples)
    packet_count = np.random.randint(10, 1000, n_samples)
    protocol_type = np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples)
    
    # Map protocol type to numerical values for classification
    protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
    protocol_numerical = np.array([protocol_map[x] for x in protocol_type])

    # Labels (Traffic types: Web, FTP, VoIP, etc.)
    traffic_labels = np.random.choice(['Web', 'FTP', 'VoIP'], n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'Flow_Duration': flow_duration,
        'Byte_Count': byte_count,
        'Packet_Count': packet_count,
        'Protocol': protocol_numerical,
        'Traffic_Type': traffic_labels
    })
    
    return df

# Load synthetic data
df = generate_synthetic_data()

# Feature engineering - extract features and labels
X = df[['Flow_Duration', 'Byte_Count', 'Packet_Count', 'Protocol']]  # Features
y = df['Traffic_Type']  # Target labels

# Encode labels (Traffic types: Web, FTP, VoIP)
y = y.map({'Web': 0, 'FTP': 1, 'VoIP': 2})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Web', 'FTP', 'VoIP'], yticklabels=['Web', 'FTP', 'VoIP'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Web', 'FTP', 'VoIP']))

# Feature importance
features = ['Flow_Duration', 'Byte_Count', 'Packet_Count', 'Protocol']
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
