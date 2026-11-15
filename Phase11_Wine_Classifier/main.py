from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. Load dataset (built into sklearn, no download!)
# ============================================================
wine = load_wine()
X, y = wine.data, wine.target

print(f"Dataset loaded: {X.shape}")
print(f"Features: {wine.feature_names}")
print(f"Classes: {wine.target_names}")
print(f"Samples per class: {[sum(y == i) for i in range(3)]}")


# ============================================================
# 2. Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")


# ============================================================
# 3. Normalize
# ============================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Features normalized")


# ============================================================
# 4. Train Random Forest (different from Logistic Regression!)
# ============================================================
model = RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    random_state=42
)
model.fit(X_train, y_train)
print("\n✅ Random Forest trained (100 trees)")


# ============================================================
# 5. Evaluate
# ============================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Results ===")
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 6. Feature importance (Random Forest specialty!)
# ============================================================
print("\n=== Feature Importance ===")
importances = model.feature_importances_
indices = importances.argsort()[::-1]

for i in indices[:5]:  # Top 5 most important
    print(f"  {wine.feature_names[i]}: {importances[i]:.4f}")


# ============================================================
# 7. Test on new wine
# ============================================================
print("\n=== Testing on new wine ===")

# First wine from each class (real examples)
test_wines = [
    X_test[0],   # class 0 (cultivar 1)
    X_test[10],  # class 1 or 2
    X_test[-1],  # last test sample
]

for i, wine_features in enumerate(test_wines):
    scaled = scaler.transform([wine_features])
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][pred]
    print(f"\nWine {i+1}:")
    print(f"  → Predicted: {wine.target_names[pred]} ({prob:.2%} confidence)")