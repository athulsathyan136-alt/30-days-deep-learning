
import pandas as pd
import numpy as np
import re
import os
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. Download dataset (small ~200KB)
# ============================================================
DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

csv_path = os.path.join(DATA_DIR, "sms.tsv")
if not os.path.exists(csv_path):
    print("Downloading SMS Spam dataset...")
    urllib.request.urlretrieve(DATA_URL, csv_path)

# Load dataset
df = pd.read_csv(csv_path, sep="\t", header=None, names=["label", "text"])
print(f"Dataset: {len(df)} messages")
print(f"Class distribution:\n{df['label'].value_counts()}")


# ============================================================
# 2. Text preprocessing
# ============================================================
def preprocess(text):
    """Clean SMS text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["text"].apply(preprocess)
df["label_bin"] = (df["label"] == "spam").astype(int)

print(f"\nSpam: {df['label_bin'].sum()} | Ham: {len(df) - df['label_bin'].sum()}")


# ============================================================
# 3. Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"].values,
    df["label_bin"].values,
    test_size=0.2,
    random_state=42,
    stratify=df["label_bin"].values
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")


# ============================================================
# 4. Vectorization — Compare Count vs TF-IDF!
# ============================================================
vectorizers = {
    "Count (BoW)": CountVectorizer(max_features=3000, stop_words="english"),
    "TF-IDF":      TfidfVectorizer(max_features=3000, stop_words="english"),
}


# ============================================================
# 5. Train Naive Bayes + Logistic Regression (compare!)
# ============================================================
print(f"\n{'=' * 70}")
print(f"{'Vectorizer':<15} {'Classifier':<20} {'Accuracy':<10}")
print(f"{'=' * 70}")

results = {}

for vec_name, vec in vectorizers.items():
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    classifiers = {
        "Naive Bayes":          MultinomialNB(),
        "Logistic Regression":  LogisticRegression(max_iter=1000),
    }
    
    for clf_name, clf in classifiers.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        
        results[(vec_name, clf_name)] = (clf, vec, y_pred, acc)
        print(f"{vec_name:<15} {clf_name:<20} {acc:.4f}")

print(f"{'=' * 70}")


# ============================================================
# 6. Best model details
# ============================================================
best_key = max(results, key=lambda k: results[k][3])
best_clf, best_vec, best_pred, best_acc = results[best_key]
print(f"\n🏆 Best: {best_key[0]} + {best_key[1]} (Accuracy: {best_acc:.4f})")

print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=["Ham", "Spam"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(f"  True Ham:      {cm[0,0]}")
print(f"  False Spam:    {cm[0,1]}")
print(f"  False Ham:     {cm[1,0]}")
print(f"  True Spam:     {cm[1,1]}")


# ============================================================
# 7. Test on new messages
# ============================================================
print(f"\n{'=' * 60}")
print("Testing on new messages")
print('=' * 60)

test_messages = [
    "Congratulations! You've won a FREE iPhone! Click here!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account will be suspended. Verify now!",
    "Thanks for the update, see you soon.",
    "Win $1,000,000 now! Reply YES to claim!",
]

for msg in test_messages:
    clean = preprocess(msg)
    vec = best_vec.transform([clean])
    pred = best_clf.predict(vec)[0]
    prob = best_clf.predict_proba(vec)[0][1]
    label = "SPAM" if pred == 1 else "HAM"
    print(f"\n{msg}")
    print(f"  → {label} ({prob:.2%} spam probability)")