# Zepto Data & AI Platform Capstone

A single repository containing the three required modules:
- `data_pipeline` — scraping, cleaning, currency conversion, SQLite, SQL + pandas
- `analytics` — Titanic profiling, EDA, classification, imbalance handling, tuning, regression
- `support_assistant` — local embeddings + ChromaDB, LangGraph routing, Pydantic output, FastAPI

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The project uses one consolidated `requirements.txt`.

## Run

### 1. Data pipeline

```bash
cd data_pipeline
python pipeline.py
```

This scrapes at least 60 books from BooksToScrape, cleans the fields, converts GBP to INR using the required fixed rate of `1 GBP = 105.50 INR`, creates `zepto_books.sqlite`, runs the required SQL queries, and demonstrates `pd.read_sql` plus `pd.merge`.

### 2. Analytics

```bash
cd analytics
python analytics_pipeline.py
```

The script loads the Titanic dataset once from Seaborn, saves `titanic.csv` as the offline fallback, performs EDA, creates charts, trains three classifiers, compares imbalance strategies, tunes Random Forest, performs fare regression, and saves the complete fitted classification pipeline as `best_pipeline.joblib`.

If the first Seaborn load cannot reach the internet, use a previously generated committed `analytics/titanic.csv`; the modeling portion can then read that CSV.

### 3. Support assistant

```bash
cd support_assistant
python ingest.py
python main.py
```

Run the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

Example:

```bash
curl -X POST http://127.0.0.1:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the delivery fee below INR 149?\"}"
```

Default `MOCK_LLM` is `1`, which is the required deterministic, fully offline grading path.

## Design decisions

### Data pipeline
The pipeline uses requests + BeautifulSoup, explicit parsing functions, a fixed project currency rate, and a normalized `categories`/`books` SQLite schema. Unexpected numeric parsing failures use median imputation; rows with unrecoverable required categorical fields are dropped with a logged reason.

### Analytics
The raw Titanic data is loaded once and immediately saved to `titanic.csv`. EDA cleaning follows the assignment's missing-value thresholds. Modeling uses a stratified split and a scikit-learn ColumnTransformer/Pipeline so preprocessing is fit only on training data.

### Support assistant
The eight supplied policy documents are stored as text files, embedded locally with `all-MiniLM-L6-v2`, and indexed in ChromaDB. LangGraph routes policy questions through retrieval and general questions directly. Only generation changes with `MOCK_LLM`; retrieval remains real in both modes.

## Git workflow requirement

Create a feature branch, make at least two commits on it, and merge it into `main`. Verify with:

```bash
git log --graph --oneline --all
```

The final submission is one public GitHub repository URL.
