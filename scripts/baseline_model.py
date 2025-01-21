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
from sklearn.model_selection import cross_val_score, train_test_split


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
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})

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

    # 数値列のみを対象にして欠損値を補完
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

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
    # 数値列のみを対象にして欠損値を補完
    num_cols = X_test.select_dtypes(include=[np.number]).columns
    X_test[num_cols] = X_test[num_cols].fillna(X_test[num_cols].mean())

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

    # データの前処理
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

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
