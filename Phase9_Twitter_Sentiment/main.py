
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. Sample tweets (built-in, no download!)
# ============================================================
# Real-world tweets with sentiment labels
tweets = [
    # Negative (0)
    ("This flight was terrible. Delayed 4 hours and no apology.", 0),
    ("Worst airline ever. Lost my luggage and customer service is useless.", 0),
    ("Never flying with you again. Horrible experience.", 0),
    ("Flight cancelled, no rebooking, stuck at airport all night.", 0),
    ("Staff was rude and unhelpful. Terrible service.", 0),
    ("My seat was broken and they refused to move me.", 0),
    ("Baggage lost for 3 days. Unacceptable.", 0),
    ("Waited 6 hours for a delayed flight. Never again.", 0),
    ("Overbooked flight, they bumped me off. Disgusting.", 0),
    ("Food was inedible, seats uncomfortable. Awful.", 0),
    ("The airline messed up my reservation completely.", 0),
    ("Rude gate agent. Complete lack of professionalism.", 0),
    ("Flight delayed, missed connection, no compensation.", 0),
    ("Charged me twice for baggage. Terrible.", 0),
    ("No legroom, no entertainment, no water. Sad.", 0),
    ("Cancelled at the last minute. Frustrating.", 0),
    ("Horrible experience. Will never recommend.", 0),
    ("They lost my bags and won't respond to emails.", 0),
    ("Rude staff and dirty planes. Yuck.", 0),
    ("Terrible customer service, terrible airline.", 0),
    
    # Neutral (1)
    ("Flight departed on time. Nothing special.", 1),
    ("Just landed. Normal flight.", 1),
    ("Standard economy class. It is what it is.", 1),
    ("Flight was okay. No complaints, no praise.", 1),
    ("Onboard WiFi didn't work but flight was fine.", 1),
    ("Average service, average food.", 1),
    ("Arrived at destination. That's what matters.", 1),
    ("It was a routine flight.", 1),
    ("Nothing memorable about the trip.", 1),
    ("Some delays, but eventually got there.", 1),
    ("Seat was fine, food was fine. Fine overall.", 1),
    ("Standard domestic flight. Nothing to report.", 1),
    ("Flight took off and landed. Basic service.", 1),
    ("Typical airline experience. Not bad, not great.", 1),
    ("Okay I guess. Nothing special.", 1),
    ("Waited a bit but eventually boarded.", 1),
    ("Everything was normal. No issues.", 1),
    ("It was a flight. Nothing more.", 1),
    ("Neither bad nor good. Just average.", 1),
    ("Got from point A to point B. Mission accomplished.", 1),
    
    # Positive (2)
    ("Amazing flight! Crew was friendly and food was great.", 2),
    ("Best airline experience ever. Highly recommend!", 2),
    ("Very comfortable seats and great entertainment.", 2),
    ("The staff went above and beyond to help.", 2),
    ("Loved the service. Will definitely fly again.", 2),
    ("Smooth boarding, on-time departure, excellent.", 2),
    ("Great legroom and delicious food.", 2),
    ("Fantastic service and friendly crew.", 2),
    ("Best flight I have ever taken. Wonderful.", 2),
    ("Superb experience from start to finish.", 2),
    ("Really enjoyed the flight. Everything was perfect.", 2),
    ("Great value for money. Excellent service.", 2),
    ("Crew was incredibly helpful and kind.", 2),
    ("WiFi was fast and entertainment was top-notch.", 2),
    ("Comfortable and enjoyable. Will fly again.", 2),
    ("Top-quality service. Impressed.", 2),
    ("Perfect flight. No complaints at all.", 2),
    ("Delightful experience. Love this airline.", 2),
    ("Fantastic! Everything about this flight was great.", 2),
    ("Outstanding service. Five stars!", 2),
]

df = pd.DataFrame(tweets, columns=["text", "label"])
print(f"Dataset: {len(df)} tweets")
print(f"Class distribution:\n{df['label'].value_counts()}")


# ============================================================
# 2. Text preprocessing
# ============================================================
def preprocess_tweet(text):
    """Clean tweet text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", lambda m: " " + m.group()[1:], text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("\nPreprocessing tweets...")
df["clean_text"] = df["text"].apply(preprocess_tweet)
df = df[df["clean_text"].str.len() > 0]
print(f"After cleaning: {len(df)} tweets")


# ============================================================
# 3. Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"].values,
    df["label"].values,
    test_size=0.2,
    random_state=42,
    stratify=df["label"].values
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")


# ============================================================
# 4. TF-IDF vectorization
# ============================================================
print("\nVectorizing text (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=1
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Vectorized shape: {X_train_vec.shape}")


# ============================================================
# 5. Train two classifiers
# ============================================================
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "Multinomial NB":       MultinomialNB(),
}

results = {}

for name, clf in classifiers.items():
    print(f"\n=== Training {name} ===")
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    results[name] = (clf, y_pred, acc)
    print(f"Accuracy: {acc:.4f}")


# ============================================================
# 6. Best model details
# ============================================================
best_name = max(results, key=lambda k: results[k][2])
best_clf, best_pred, best_acc = results[best_name]

print(f"\n{'=' * 60}")
print(f"Best model: {best_name} ({best_acc:.4f})")
print('=' * 60)

print("\nClassification Report:")
print(classification_report(
    y_test, best_pred,
    target_names=["Negative", "Neutral", "Positive"],
    zero_division=0
))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(f"              Predicted")
print(f"              Neg  Neu  Pos")
labels = ["Neg", "Neu", "Pos"]
for i, row in enumerate(cm):
    print(f"Actual {labels[i]}:  {row}")


# ============================================================
# 7. Test on new tweets
# ============================================================
print(f"\n{'=' * 60}")
print(f"Testing on new tweets (using {best_name})")
print('=' * 60)

test_tweets = [
    "I absolutely love this new phone! Best purchase ever!",
    "Worst customer service ever. Never coming back.",
    "Just had lunch. It was okay.",
    "The weather today is amazing, perfect for a walk!",
    "Ugh, stuck in traffic again. So frustrating.",
]

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

for tweet in test_tweets:
    clean = preprocess_tweet(tweet)
    vec = vectorizer.transform([clean])
    pred = best_clf.predict(vec)[0]
    prob = best_clf.predict_proba(vec)[0]
    
    print(f"\nTweet: {tweet}")
    print(f"  → {label_map[pred]} (confidence: {prob[pred]:.2%})")