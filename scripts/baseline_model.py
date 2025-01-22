import datetime
from typing import Dict, List, Tuple

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def load_data(
    train_path: str, test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    データを読み込みます。

    Args:
        train_path (str): 訓練データのパス
        test_path (str): テストデータのパス

    Returns:
        tuple: 訓練データとテストデータのタプル
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    データの前処理を行います。

    Args:
        X (pd.DataFrame): 特徴量データフレーム

    Returns:
        pd.DataFrame: 前処理後の特徴量データフレーム
    """
    # Embarkedの欠損値補完
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # Fareの欠損値補完
    for i, row in X[X["Fare"].isnull()].iterrows():
        median_fare = X[
            (X["Pclass"] == row["Pclass"])
            & (X["Embarked"] == row["Embarked"])
            & (X["SibSp"] == row["SibSp"])
            & (X["Parch"] == row["Parch"])
        ]["Fare"].median()
        X.at[i, "Fare"] = median_fare

    # FareGroupの作成
    X["FareGroup"] = pd.qcut(X["Fare"], 5, labels=False)

    # TitleGroupの作成
    X["Title"] = X["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Other",
        "Rev": "Other",
        "Col": "Other",
        "Major": "Other",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Lady": "Other",
        "Sir": "Other",
        "Mme": "Mrs",
        "Capt": "Other",
        "Countess": "Other",
        "Jonkheer": "Other",
        "Don": "Other",
        "Dona": "Other",
    }
    X["TitleGroup"] = X["Title"].map(title_mapping)

    # Ageの欠損値補完
    for pclass in X["Pclass"].unique():
        for title in X["TitleGroup"].unique():
            median_age = X[
                (X["Pclass"] == pclass)
                & (X["TitleGroup"] == title)
                & (X["Age"].notnull())
            ]["Age"].median()
            X.loc[
                (X["Pclass"] == pclass)
                & (X["TitleGroup"] == title)
                & (X["Age"].isnull()),
                "Age",
            ] = median_age

    # AgeGroupの作成
    X["AgeGroup"] = pd.qcut(X["Age"], 6, labels=False)

    # Cabinの欠損値補完
    for i, row in X[X["Cabin"].isnull()].iterrows():
        same_ticket_records = X[X["Ticket"] == row["Ticket"]]
        if not same_ticket_records[
            same_ticket_records["Cabin"].notnull()
        ].empty:
            most_frequent_cabin = same_ticket_records["Cabin"].mode()[0]
            X.at[i, "Cabin"] = most_frequent_cabin
        else:
            X.at[i, "Cabin"] = "Z"

    # CabinKeyの作成
    X["CabinKey"] = X["Cabin"].apply(lambda x: x[:2] + str(len(x)))

    # FamilyIDの作成
    X["FamilyName"] = X["Name"].apply(lambda name: name.split(",")[0])
    X["FamilyID"] = X.groupby(["FamilyName", "Embarked", "Pclass", "Cabin"])[
        "PassengerId"
    ].transform("min")

    # FamilyScaleの作成
    X["FamilySize"] = X.groupby("FamilyID")["PassengerId"].transform("count")
    X["FamilyScale"] = X["FamilySize"].apply(
        lambda x: 0 if x == 1 else (1 if 2 <= x <= 4 else 2)
    )

    # 数値列のみを対象にして欠損値を補完
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

    # カテゴリ変数を数値に変換
    le = LabelEncoder()
    for col in ["Sex", "Embarked", "TitleGroup"]:
        X[col] = le.fit_transform(X[col])

    return X


def optimize_hyperparameters(
    X: pd.DataFrame, y: pd.Series, cfg: DictConfig
) -> Dict[str, float]:
    """
    ハイパーパラメータの最適化を行います。

    Args:
        X (pd.DataFrame): 特徴量データフレーム
        y (pd.Series): ターゲットデータ
        cfg (DictConfig): コンフィグデータ

    Returns:
        dict: 最適化されたハイパーパラメータ
    """

    def objective(params: Dict[str, float]) -> float:
        clf = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            random_state=cfg.split.random_state,
        )
        score = cross_val_score(
            clf, X, y, cv=cfg.model.cv, scoring="accuracy"
        ).mean()
        return -score

    search_space = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 200, 10)),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 30, 1)),
    }

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=cfg.model.max_evals,
        trials=trials,
    )

    if best_params is None:
        best_params = {}

    best_params = {k: float(v) for k, v in best_params.items()}

    return best_params


def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict[str, float],
    random_state: int,
    cv: int,
) -> Tuple[RandomForestClassifier, float]:
    """
    モデルの訓練と評価を行います。

    Args:
        X (pd.DataFrame): 特徴量データフレーム
        y (pd.Series): ターゲットデータ
        best_params (dict): 最適化されたハイパーパラメータ
        random_state (int): 乱数シード
        cv (int): クロスバリデーションの分割数

    Returns:
        tuple: 訓練されたモデルと評価スコア
    """
    model = RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        random_state=random_state,
    )
    model.fit(X, y)
    accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
    return model, accuracy


def log_results(
    best_params: Dict[str, float],
    accuracy: float,
    features: List[str],
    model: RandomForestClassifier,
) -> None:
    """
    結果をMLflowにログとして記録します。

    Args:
        best_params (dict): 最適化されたハイパーパラメータ
        accuracy (float): 評価スコア
        features (list): 使用した特徴量のリスト
        model (sklearn.base.BaseEstimator): 訓練されたモデル
    """
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("features", features)
    mlflow.sklearn.log_model(model, "model")


def save_predictions(
    test_data: pd.DataFrame,
    X_test: pd.DataFrame,
    model: RandomForestClassifier,
    current_time: str,
) -> None:
    """
    テストデータに対する予測結果を保存します。

    Args:
        test_data (pd.DataFrame): テストデータ
        X_test (pd.DataFrame): 前処理後のテストデータの特徴量
        model (sklearn.base.BaseEstimator): 訓練されたモデル
        current_time (str): 現在の時刻を表す文字列
    """
    predictions = model.predict(X_test)
    output_path = f"./results/{current_time}.csv"
    output = pd.DataFrame(
        {"PassengerId": test_data["PassengerId"], "Perished": predictions}
    )
    output.to_csv(output_path, index=False)
    print(f"Prediction results saved to {output_path}")


@hydra.main(config_path="../scripts", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    メイン関数。データの読み込み、前処理、ハイパーパラメータの最適化、モデルの訓練と評価、結果のログ記録、予測結果の保存を行います。

    Args:
        cfg (DictConfig): コンフィグデータ
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = current_time
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()

    # データの読み込み
    train_data, test_data = load_data(cfg.data.train_path, cfg.data.test_path)

    # データの前処理と特徴量エンジニアリング
    combined_data = pd.concat([train_data, test_data], sort=False)
    combined_data = preprocess_data(combined_data)
    train_data = combined_data[
        combined_data["PassengerId"].isin(train_data["PassengerId"])
    ]
    test_data = combined_data[
        combined_data["PassengerId"].isin(test_data["PassengerId"])
    ]

    # 特徴量とターゲットに分割
    X = train_data[cfg.features]
    y = train_data[cfg.target]

    # ハイパーパラメータの最適化
    best_params = optimize_hyperparameters(X, y, cfg)

    # モデルの訓練と評価
    model, accuracy = train_and_evaluate_model(
        X, y, best_params, cfg.split.random_state, cfg.model.cv
    )

    # MLflowにログを記録
    log_results(best_params, accuracy, cfg.features, model)

    # テストデータの前処理
    X_test = test_data[cfg.features]

    # 予測結果の保存
    save_predictions(test_data, X_test, model, current_time)

    mlflow.end_run()


if __name__ == "__main__":
    main()
