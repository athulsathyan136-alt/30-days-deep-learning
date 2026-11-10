
import pandas as pd
import urllib.request
import zipfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. Download dataset
# ============================================================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

zip_path = os.path.join(DATA_DIR, "smsspam.zip")
if not os.path.exists(zip_path):
    print("Downloading SMS Spam dataset...")
    urllib.request.urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

# Load the SMS Spam dataset
df = pd.read_csv(
    os.path.join(DATA_DIR, "SMSSpamCollection"),
    sep="\t",
    names=["label", "text"],
    encoding="utf-8"
)

print(f"Dataset loaded: {len(df)} messages")
print(f"Classes: {df['label'].value_counts().to_dict()}")


# ============================================================
# 2. Prepare data
# ============================================================
X = df["text"].values
y = (df["label"] == "spam").astype(int).values   # 1 = spam, 0 = ham

print(f"\nSpam: {y.sum()} | Ham: {len(y) - y.sum()}")


# ============================================================
# 3. Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")


# ============================================================
# 4. TF-IDF Vectorization
# ============================================================
print("\nVectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vectorized shape: {X_train_vec.shape}")


# ============================================================
# 5. Train Logistic Regression
# ============================================================
print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ============================================================
# 6. Evaluate
# ============================================================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Results ===")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

print(f"Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Ham:  {cm[0,0]}    False Spam: {cm[0,1]}")
print(f"  False Ham: {cm[1,0]}    True Spam:  {cm[1,1]}")


# ============================================================
# 7. Test on new messages
# ============================================================
print("\n=== Testing on new messages ===")

test_messages = [
    "Congratulations! You've won a FREE iPhone! Click here to claim.",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account will be suspended. Verify now: bit.ly/xyz",
    "Thanks for the update, I'll review it tomorrow.",
    "You have won $1,000,000! Reply YES to claim your prize.",
]

for msg in test_messages:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]
    label = "SPAM" if pred == 1 else "HAM"
    print(f"\nMessage: {msg}")
    print(f"  → {label} ({prob:.2%} spam probability)")