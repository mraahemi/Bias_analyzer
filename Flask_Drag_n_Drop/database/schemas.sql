DROP TABLE IF EXISTS dim_models;
DROP TABLE IF EXISTS fct_data;

CREATE TABLE dim_models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR2(50),
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    uploaded_flag INTEGER
);

CREATE TABLE fct_data (
    FOREIGN_KEY(model_id) REFERENCES dim_models(model_id),
    column_name VARCHAR2(100),
    column_value VARCHAR2(400),
    target_variable INTEGER,
    donated_flag INTEGER
);

