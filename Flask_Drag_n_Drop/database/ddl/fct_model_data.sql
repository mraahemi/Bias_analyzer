DROP TABLE IF EXISTS fct_model_data ;

--todo: create model_id as a foreign_key

CREATE TABLE fct_model_data (
    model_id INTEGER,
    column_name VARCHAR2(100),
    column_value VARCHAR2(400),
    is_target_variable INTEGER,
    is_donated INTEGER --,
    --FOREIGN_KEY(model_id) REFERENCES dim_models(model_id),
)

