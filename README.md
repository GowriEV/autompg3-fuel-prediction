# AutoMPG Fuel Intelligence

## Run locally
1. Install Python 3.10–3.13.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended):
   - Windows: `py -m venv .venv` then `.venv\Scripts\activate`
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Run the app: `python -m streamlit run app.py`
6. Open the local URL printed in the terminal.

The app compares Linear Regression, Random Forest, and Gradient Boosting using a held-out test split. It imputes missing values, scales numeric features, and one-hot encodes vehicle origin inside a pipeline. Upload a CSV from the sidebar or use the included `auto-mpg.csv`.
