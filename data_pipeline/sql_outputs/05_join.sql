SELECT c.category_name, b.title, b.rating, b.price_gbp
                      FROM books b JOIN categories c ON b.category_id=c.category_id
                      ORDER BY b.rating DESC, c.category_name, b.title LIMIT 10;