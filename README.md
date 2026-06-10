
# Recommendation System

FastAPI-сервис рекомендаций, который ранжирует посты для пользователей с использованием CatBoost-моделей, пользовательских и постовых признаков, а также embedding-based similarity signals.

Проект включает A/B testing setup с отдельными control и test моделями. Control model использует классические признаки для ранжирования, а test model дополнительно использует similarity features на основе post embeddings и user interaction embeddings.

## Features

- FastAPI-сервис для real-time рекомендаций постов
- Ranking models на основе CatBoost
- Hash-based A/B test split между control и test группами
- Загрузка user, post и user-topic features из PostgreSQL
- Embedding similarity features с использованием dot product и cosine similarity
- Настраиваемый лимит рекомендаций
- Pydantic response schemas

## Tech Stack

- Python
- FastAPI
- CatBoost
- pandas
- NumPy
- SQLAlchemy
- PostgreSQL
- Pydantic
- Uvicorn

## API

### Получение рекомендаций постов

```http
GET /post/recommendations/?id={user_id}&time={datetime}&limit={limit}
```

Example response:

```json
{
  "exp_group": "test",
  "recommendations": [
    {
      "id": 123,
      "text": "Post text",
      "topic": "technology"
    }
  ]
}
```

## How it works

1. Сервис назначает пользователя в control или test группу с помощью deterministic hash-based split.
2. User, post и user-topic features загружаются из PostgreSQL.
3. Candidate posts дополняются пользовательскими и topic-level признаками.
4. Control group ранжируется с помощью baseline CatBoost model.
5. Test group ранжируется с помощью CatBoost model, улучшенной embedding similarity features.
6. Top-ranked posts возвращаются через API.

## Project Structure

```text
.
├── app.py                  # FastAPI application and recommendation logic
├── schema.py               # Pydantic response models
├── requirements.txt        # Project dependencies
└── dl_model_training.ipynb.ipynb
```

## Models

Проект использует две CatBoost-модели:

- `model_control.cbm` — baseline ranking model
- `model_test.cbm` — model with additional embedding-based similarity features

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```

## Notes

Этот проект демонстрирует production-style recommendation API с model-based ranking, feature enrichment и A/B testing logic.
````
