import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# データの読み込み
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

# 特徴量とターゲットの選択
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train_data[features]
y = train_data["Perished"]

# 前処理
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})
X = X.fillna(X.mean())

# モデルの訓練
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 検証データでの評価
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred)}")

# テストデータの前処理
X_test = test_data[features]
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})
X_test["Embarked"] = X_test["Embarked"].map({"S": 0, "C": 1, "Q": 2})
X_test = X_test.fillna(X_test.mean())

# 予測
predictions = model.predict(X_test)

# 結果の保存
output = pd.DataFrame(
    {"PassengerId": test_data["PassengerId"], "Perished": predictions}
)
output.to_csv("./results/baseline_predictions.csv", index=False)
print("Prediction results saved to ./results/baseline_predictions.csv")
