drop table if exists file_contents;

CREATE TABLE file_contents (
   id INTEGER NOT NULL,
   name VARCHAR(300),
   modeldata VARCHAR(300),
   data BLOB,
   PRIMARY KEY (id)
)