# Data Pipeline

The module implements scrape → clean → convert → relational store → SQL → pandas.

- Source: BooksToScrape
- Scope: first available categories until at least 60 books across at least 3 categories
- Currency: fixed project rate `1 GBP = 105.50 INR`
- Numeric parsing failures: median imputation
- Required categorical failures: row dropped rather than crashing
- Database: SQLite with `categories` and `books` linked by PK/FK
- Query outputs: `sql_outputs/`
