# Crop Recommendation System (CSE_16)

A small supervised learning project that uses a scaled K-Nearest Neighbors (KNN) classifier to recommend a crop from seven soil and climate measurements. The Streamlit interface reads a saved model; it never silently trains or fabricates results at startup.

## Problem and features

Farmers need to interpret soil nutrient values alongside local climate conditions when choosing crops. This educational application demonstrates a data-driven recommendation based on Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH, and rainfall. It includes CSV quality checks, stratified 80:20 splitting, a training-only StandardScaler in a scikit-learn Pipeline, cross-validation over K = 3, 5, 7, 9, 11, held-out evaluation, plots, and a multi-page dashboard.

## Dataset

Download the real [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) and place `Crop_recommendation.csv` in `dataset/`. No training data is bundled or generated. The expected columns are `N, P, K, temperature, humidity, ph, rainfall, label`; the dataset is commonly described as roughly 2,200 rows across 22 crop classes. The trainer reports the actual observed counts and rejects missing columns, missing values, non-numeric feature values, and invalid labels.

## Technologies

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib, and ReportLab.

## Structure

```text
dataset/                 Place Crop_recommendation.csv here
notebooks/               Exploratory analysis notebook
models/                  Trained pipeline (created by training)
reports/                 Academic report and actual model results
  figures/               Evaluation plots
screenshots/             Add screenshots after launching the app
src/                     Reusable preprocessing, training, prediction, evaluation
app.py                   Streamlit dashboard
requirements.txt
```

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

After putting the CSV in `dataset/`:

```powershell
python -m src.train_model
python -m src.evaluate_model
streamlit run app.py
```

Training selects K by stratified cross-validation on the training partition only. The final fitted pipeline is saved at `models/knn_model.pkl`; evaluation metrics and plots are generated from the held-out test partition at `reports/model_results.json` and `reports/figures/`. Evaluation also refreshes the academic PDF report. To regenerate the report separately, run `python -m src.generate_report`. Paths are resolved relative to the repository, so commands can be run from its root.

## Evaluation and screenshots

No accuracy, precision, recall, F1 score, confusion matrix, or screenshot is supplied as a result until the real dataset has been used and the application run. The report template at `reports/Crop_Recommendation_System_Report.pdf` explicitly marks these sections as pending. Once the app is running, capture a real screenshot and save it under `screenshots/` to include it in a later report generation.

## Input ranges and limitations

The app validates that entries are finite numbers, and presents training-data observed ranges when available. It does not assert that a model recommendation is agronomic advice. Results depend on dataset coverage and quality, and the dataset may not represent local soils, cultivars, irrigation, market conditions, or changing weather. KNN uses scaled Euclidean distance and is limited to the 22 labels represented in its training data.

## Future scope

Add region-specific validated data, explain recommendations in agronomic context, support model monitoring and data versioning, and evaluate carefully documented alternatives while retaining KNN as the project’s required primary algorithm.
