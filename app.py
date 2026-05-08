import os
import hashlib
import pickle
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from sqlalchemy import create_engine
from catboost import CatBoostClassifier, Pool

from schema import PostGet, Response


SALT = "my_salt"
CONTROL_PERCENT = 50

app = FastAPI()


def get_exp_group(user_id: int) -> str:
    value = int(
        hashlib.md5(f"{user_id}_{SALT}".encode()).hexdigest(),
        16
    )
    return "control" if value % 100 < CONTROL_PERCENT else "test"



def get_model_path(model_name: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        return f"/workdir/user_input/{model_name}.cbm"

    return f"{model_name}.cbm"

def load_control_model():
    path = get_model_path("model_control")

    if os.environ.get("IS_LMS") != "1" and path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

    model = CatBoostClassifier()
    model.load_model(path)
    return model


def load_test_model():
    path = get_model_path("model_test")

    model = CatBoostClassifier()
    model.load_model(path)
    return model


model_control = load_control_model()
model_test = load_test_model()

def batch_load_sql(query: str) -> pd.DataFrame:
    chunksize = 200000

    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )

    conn = engine.connect().execution_options(stream_results=True)
    chunks = []

    for chunk_df in pd.read_sql(query, conn, chunksize=chunksize):
        chunks.append(chunk_df)

    conn.close()

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def load_features():
    return batch_load_sql("""
        SELECT *
        FROM irina_azarova_features_lesson_24_users
    """)


def load_posts():
    return batch_load_sql("""
        SELECT *
        FROM irina_azarova_posts_features_lesson_24
    """)


def load_user_topic_features():
    return batch_load_sql("""
        SELECT *
        FROM irina_azarova_user_topic_features_lesson_24
    """)


user_features = load_features()
posts = load_posts()
user_topic_stats = load_user_topic_features()


EMB_COLS = [col for col in posts.columns if col.startswith("emb_")]
MODEL_FEATURES_CONTROL = model_control.feature_names_
MODEL_FEATURES_TEST = model_test.feature_names_

CAT_COLS = ["country", "os", "source", "topic"]


def add_similarity_features(df: pd.DataFrame, emb_cols: List[str]) -> pd.DataFrame:
    df = df.copy()

    post_matrix = df[emb_cols].values

    user_view_cols = [f"user_view_{col}" for col in emb_cols]
    user_like_cols = [f"user_like_{col}" for col in emb_cols]

    user_view_matrix = df[user_view_cols].fillna(0).values
    user_like_matrix = df[user_like_cols].fillna(0).values

    df["sim_view_dot"] = np.sum(post_matrix * user_view_matrix, axis=1)
    df["sim_like_dot"] = np.sum(post_matrix * user_like_matrix, axis=1)

    post_norm = np.linalg.norm(post_matrix, axis=1) + 1e-8
    user_view_norm = np.linalg.norm(user_view_matrix, axis=1) + 1e-8
    user_like_norm = np.linalg.norm(user_like_matrix, axis=1) + 1e-8

    df["sim_view_cos"] = df["sim_view_dot"] / (post_norm * user_view_norm)
    df["sim_like_cos"] = df["sim_like_dot"] / (post_norm * user_like_norm)

    return df


def recommend_control(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    X = df[MODEL_FEATURES_CONTROL]
    cat_cols = [col for col in CAT_COLS if col in X.columns]

    pool = Pool(X, cat_features=cat_cols)
    df["score"] = model_control.predict_proba(pool)[:, 1]

    return df


def recommend_test(df: pd.DataFrame) -> pd.DataFrame:
    df = add_similarity_features(df, EMB_COLS)

    X = df[MODEL_FEATURES_TEST]
    cat_cols = [col for col in CAT_COLS if col in X.columns]

    pool = Pool(X, cat_features=cat_cols)
    df["score"] = model_test.predict_proba(pool)[:, 1]

    return df


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 5):
    exp_group = get_exp_group(id)

    user_row = user_features[user_features["user_id"] == id]

    if user_row.empty:
        return {
            "exp_group": exp_group,
            "recommendations": []
        }

    user_row = user_row.drop(columns=["user_id"])

    df = posts.copy()

    for col in user_row.columns:
        df[col] = user_row.iloc[0][col]


    user_topic_row = user_topic_stats[user_topic_stats["user_id"] == id]

    df = df.merge(
        user_topic_row.drop(columns=["user_id"], errors="ignore"),
        on="topic",
        how="left"
    )

    cat_cols_in_df = [col for col in CAT_COLS if col in df.columns]
    num_cols_in_df = [
        col for col in df.columns
        if col not in cat_cols_in_df + ["post_id", "text"]
    ]

    df[cat_cols_in_df] = df[cat_cols_in_df].fillna("unknown")
    df[num_cols_in_df] = df[num_cols_in_df].fillna(0)

    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(float)

    if exp_group == "control":
        df = recommend_control(df)
    elif exp_group == "test":
        df = recommend_test(df)
    else:
        raise ValueError("unknown group")

    top_posts = df.sort_values("score", ascending=False).head(limit)

    return {
        "exp_group": exp_group,
        "recommendations": [
            PostGet(
                id=row.post_id,
                text=row.text,
                topic=row.topic,
            )
            for row in top_posts.itertuples()
        ]
    }