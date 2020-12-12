DROP TABLE IF EXISTS dim_models;

CREATE TABLE dim_models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR2(50),
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_uploaded_by_user INTEGER,
    target_variable VARCHAR2(50),
    data_table VARCHAR2(50),
    model_description VARCHAR2(2000)
);
