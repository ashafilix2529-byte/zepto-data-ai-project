import sqlite3
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

BASE_URL = "https://books.toscrape.com/"
RATE_GBP_TO_INR = 105.50
DB_PATH = Path(__file__).with_name("zepto_books.sqlite")
OUTPUT_DIR = Path(__file__).with_name("sql_outputs")


def rating_to_int(text):
    mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    return mapping.get(text)


def parse_book(url, category):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    article = soup.select_one("article.product_page")
    title = soup.select_one("div.product_main h1").get_text(strip=True)
    price_text = soup.select_one("div.product_main p.price_color").get_text(strip=True)
    availability = soup.select_one("div.product_main p.instock").get_text(" ", strip=True)
    rating_text = article.select_one("p.star-rating")["class"][1]
    price = float(re.sub(r"[^0-9.]", "", price_text))
    return {
        "title": title,
        "price_gbp": price,
        "rating": rating_to_int(rating_text),
        "availability": availability,
        "in_stock": "In stock" in availability,
        "category": category,
    }


def get_category_urls():
    r = requests.get(BASE_URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.select("ul.nav-list ul li a"):
        name = a.get_text(strip=True)
        href = a.get("href")
        if href:
            links.append((name, requests.compat.urljoin(BASE_URL, href)))
    return links[:5]


def scrape(min_rows=60):
    rows = []
    for category, category_url in get_category_urls():
        page_url = category_url
        while page_url and len(rows) < min_rows:
            r = requests.get(page_url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.select("article.product_pod h3 a"):
                url = requests.compat.urljoin(page_url, a["href"])
                try:
                    rows.append(parse_book(url, category))
                except Exception as exc:
                    print(f"Skipping malformed book: {url} -> {exc}")
            next_link = soup.select_one("li.next a")
            page_url = requests.compat.urljoin(page_url, next_link["href"]) if next_link else None
            if len(rows) >= min_rows:
                break
    return pd.DataFrame(rows[:max(min_rows, len(rows))])


def clean(df):
    df = df.copy()
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if df["price_gbp"].isna().any():
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    if df["rating"].isna().any():
        df["rating"] = df["rating"].fillna(df["rating"].median()).round().astype(int)
    df = df.dropna(subset=["title", "category"]).copy()
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = df["price_gbp"] * RATE_GBP_TO_INR
    return df


def load_sqlite(df):
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("""CREATE TABLE categories(
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    )""")
    cur.execute("""CREATE TABLE books(
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY(category_id) REFERENCES categories(category_id)
    )""")
    for cat in sorted(df["category"].unique()):
        cur.execute("INSERT INTO categories(category_name) VALUES (?)", (cat,))
    cat_map = dict(cur.execute("SELECT category_name, category_id FROM categories"))
    for row in df.itertuples(index=False):
        cur.execute("""INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (row.title, row.price_gbp, row.price_inr, row.rating,
             int(row.in_stock), cat_map[row.category]))
    con.commit()
    return con


def run_queries(con):
    queries = {
        "01_select_where": "SELECT title, price_gbp FROM books WHERE price_gbp < 20;",
        "02_order_by_limit": "SELECT title, rating FROM books ORDER BY rating DESC, title LIMIT 10;",
        "03_distinct": "SELECT DISTINCT rating FROM books ORDER BY rating;",
        "04_between": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30 ORDER BY price_gbp;",
        "05_join": """SELECT c.category_name, b.title, b.rating, b.price_gbp
                      FROM books b JOIN categories c ON b.category_id=c.category_id
                      ORDER BY b.rating DESC, c.category_name, b.title LIMIT 10;"""
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, sql in queries.items():
        out = pd.read_sql(sql, con)
        (OUTPUT_DIR / f"{name}.sql").write_text(sql, encoding="utf-8")
        out.to_csv(OUTPUT_DIR / f"{name}_output.csv", index=False)
        print(f"\n--- {name} ---\n{sql}\n{out.to_string(index=False)}")
    return queries


def main():
    df = scrape(60)
    if df.empty:
        raise RuntimeError(
        "No books were scraped successfully. Check the scraping/parsing logic."
    )

    print("Scraped rows:", len(df), "categories:", df["category"].nunique())
    if len(df) < 60 or df["category"].nunique() < 3:
        raise RuntimeError("Acceptance criteria not met: need >=60 books across >=3 categories.")
    df = clean(df)
    con = load_sqlite(df)
    queries = run_queries(con)
    join_sql = queries["05_join"]
    sql_join = pd.read_sql(join_sql, con)
    books_df = pd.read_sql("SELECT * FROM books", con)
    cats_df = pd.read_sql("SELECT * FROM categories", con)
    merged = books_df.merge(cats_df, on="category_id")
    merged = merged[["category_name", "title", "rating", "price_gbp"]].sort_values(
        ["rating", "category_name", "title"], ascending=[False, True, True]
    ).head(10).reset_index(drop=True)
    sql_join = sql_join.reset_index(drop=True)
    print("\nSQL JOIN result:\n", sql_join)
    print("\npd.merge result:\n", merged)
    print("\nEquivalent:", sql_join.equals(merged))
    con.close()


if __name__ == "__main__":
    main()
