DROP TABLE IF EXISTS posts;

CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    bias TEXT NOT NULL,
    content TEXT NOT NULL
);
