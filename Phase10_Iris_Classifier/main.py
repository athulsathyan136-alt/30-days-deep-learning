
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. Load dataset (built-in, no download!)
# ============================================================
iris = load_iris()
X, y = iris.data, iris.target

print(f"Dataset loaded: {X.shape}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")
print(f"Samples per class: {[sum(y == i) for i in range(3)]}")


# ============================================================
# 2. Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")


# ============================================================
# 3. Normalize features
# ============================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Features normalized (mean=0, std=1)")


# ============================================================
# 4. Train Logistic Regression
# ============================================================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("\n✅ Model trained")


# ============================================================
# 5. Evaluate
# ============================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Results ===")
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 6. Test on new flowers
# ============================================================
print("\n=== Testing on new flowers ===")

test_flowers = [
    [5.1, 3.5, 1.4, 0.2],   # Typical setosa
    [6.0, 2.7, 5.1, 1.6],   # Typical virginica
    [5.9, 3.0, 4.2, 1.5],   # Typical versicolor
]

for flower in test_flowers:
    scaled = scaler.transform([flower])
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][pred]
    print(f"\nFlower: {flower}")
    print(f"  → Predicted: {iris.target_names[pred]} ({prob:.2%})")