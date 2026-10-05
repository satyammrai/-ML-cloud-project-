"""Train a small Iris classifier and save it to disk."""
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.joblib"


def train(model_path: Path = MODEL_PATH) -> float:
    """Train the model, save it, and return the test accuracy."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = float(clf.score(X_test, y_test))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": list(iris.target_names)}, model_path)
    return accuracy


if __name__ == "__main__":
    acc = train()
    print(f"Model trained. Test accuracy: {acc:.3f}. Saved to {MODEL_PATH}")
