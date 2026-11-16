
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


# ============================================================
# 1. Load dataset
# ============================================================
# Note: sklearn's load_diabetes is a regression dataset.
# We'll binarize the target to make it a classification problem.
diabetes = load_diabetes()
X, y_continuous = diabetes.data, diabetes.target

# Convert target to binary: 1 if diabetes progressed, 0 otherwise
y = (y_continuous > np.median(y_continuous)).astype(int)

print(f"Dataset loaded: {X.shape}")
print(f"Features: {diabetes.feature_names}")
print(f"Classes: {np.bincount(y)} → {len(y)} samples")


# ============================================================
# 2. Train/test split (stratified)
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


# ============================================================
# 4. Train THREE classifiers (compare!)
# ============================================================
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = (clf, y_pred, acc, auc)
    print(f"\n=== {name} ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc:.4f}")


# ============================================================
# 5. Detailed results for best model
# ============================================================
best_name = max(results, key=lambda k: results[k][2])
best_clf, best_pred, best_acc, best_auc = results[best_name]

print(f"\n{'=' * 60}")
print(f"Best model: {best_name} (Accuracy: {best_acc:.4f}, AUC: {best_auc:.4f})")
print('=' * 60)

print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=["No Diabetes", "Diabetes"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(f"  True Negative:  {cm[0,0]}")
print(f"  False Positive: {cm[0,1]}")
print(f"  False Negative: {cm[1,0]}")
print(f"  True Positive:  {cm[1,1]}")


# ============================================================
# 6. Feature importance
# ============================================================
if hasattr(best_clf, "feature_importances_"):
    print(f"\n=== Feature Importance ({best_name}) ===")
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in indices[:5]:
        print(f"  {diabetes.feature_names[i]}: {importances[i]:.4f}")


# ============================================================
# 7. Test on new patient
# ============================================================
print("\n=== Testing on new patients ===")

test_patients = [
    X_test[0],
    X_test[5],
    X_test[-1],
]

for i, patient in enumerate(test_patients):
    scaled = scaler.transform([patient])
    pred = best_clf.predict(scaled)[0]
    prob = best_clf.predict_proba(scaled)[0][1]
    label = "Diabetes" if pred == 1 else "No Diabetes"
    print(f"\nPatient {i+1}:")
    print(f"  → Predicted: {label} ({prob:.2%} diabetes probability)")