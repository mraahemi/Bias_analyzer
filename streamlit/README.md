#### Streamlit app

###### Installation instructions
If you have conda you can do the following otherwise skip this step
and go straight to the next or install [conda](https://docs.anaconda.com/anaconda/install/https://docs.anaconda.com/anaconda/install/)
```python
conda create -n testenv python=3.8
conda activate testenv
```
testenv is the name of the python environment and can be changed to user's preference.


go to the streamlit directory in this repository and execute
```python
pip install -r requirements.txt
```
This will download all dependencies needed to run the project. 
Then do the following.
```python
streamlit run multiple_page_agg.py
```
This will open another tab in your default browser and show you the app.