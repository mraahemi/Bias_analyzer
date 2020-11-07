#!/bin/bash

if [ -f "database/model_information.db" ]; then
    rm "database/model_information.db"
    echo "deleted database/model_information"

fi
python database/init_db.py