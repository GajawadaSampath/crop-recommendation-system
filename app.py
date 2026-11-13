"""CropWise Streamlit interface for the existing trained KNN pipeline."""
import base64
import json
from pathlib import Path

import streamlit as st

from src.prediction import MODEL_PATH, predict_crop
from src.preprocessing import FEATURES

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "reports" / "model_results.json"


def background_image(path):
    """Return an embedded local image, or a CSS fallback if it is unavailable."""
    try:
        image_data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return "none"
    return f'url("data:image/jpeg;base64,{image_data}")'


FIELD_IMAGE = background_image(ROOT / "assets" / "crop-field.jpg")
FARMERS_IMAGE = background_image(ROOT / "assets" / "farmers-field.jpg")

st.set_page_config(page_title="CropWise · Crop Recommendation", page_icon="🌱", layout="wide")

st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --forest:#123c2b; --green:#197449; --mint:#eaf5ed; --ink:#183328; --muted:#65796e; --field-image:__FIELD_IMAGE__; --farmers-image:__FARMERS_IMAGE__; }
    .stApp { background:linear-gradient(155deg,rgba(247,249,242,.22) 0%,rgba(250,250,246,.28) 48%,rgba(242,248,240,.22) 100%),var(--farmers-image); background-size:cover; background-position:center; background-attachment:fixed; color:var(--ink); }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Manrope','DM Sans',sans-serif; color:var(--forest); letter-spacing:-.025em; }
    [data-testid="stMainBlockContainer"] { background:rgba(250,250,246,.9); border:1px solid rgba(255,255,255,.72); border-radius:22px; padding:1.5rem 2rem 2rem; box-shadow:0 12px 34px rgba(24,51,40,.08); }
    [data-testid="stMarkdownContainer"], [data-testid="stText"], [data-testid="stCaptionContainer"] { color:var(--ink); }
    [data-testid="stCaptionContainer"] { color:#53665b; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#e8f3e9,#f4f8f3); border-right:1px solid #dbe9dd; }
    [data-testid="stSidebar"] h1 { font-size:1.45rem; }
    .brand-mark { display:flex; align-items:center; gap:.7rem; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.35rem; color:var(--forest); padding:.35rem 0 .8rem; }
    .brand-icon { width:2.35rem; height:2.35rem; display:grid; place-items:center; border-radius:13px; background:#d9eddd; font-size:1.3rem; }
    .hero { position:relative; isolation:isolate; overflow:hidden; padding:2.1rem 2.35rem; border-radius:24px; background:linear-gradient(rgba(15,32,23,.26),rgba(15,32,23,.26)),var(--field-image); background-size:cover; background-position:center 58%; color:#fff; box-shadow:0 16px 38px #174b3024; margin:.35rem 0 1.35rem; }
    .hero:after { content:'🌿'; position:absolute; z-index:1; right:5%; top:3%; font-size:7rem; opacity:.12; transform:rotate(-18deg); }
    .hero > * { position:relative; z-index:2; }
    .hero h1 { color:#fff!important; text-shadow:0 1px 3px rgba(0,0,0,.24); font-size:clamp(2rem,4vw,3.05rem); line-height:1.12; margin:.3rem 0 .65rem; }
    .hero p { max-width:680px; color:#f1f5f1; text-shadow:0 1px 3px rgba(0,0,0,.3); font-size:1.04rem; line-height:1.65; margin:0; }
    .eyebrow { color:#d5eadb; text-shadow:0 1px 3px rgba(0,0,0,.3); font-size:.76rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
    .hero-pills { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.15rem; }
    .pill { border:1px solid rgba(255,255,255,.58); border-radius:999px; background:rgba(15,32,23,.38); color:#fff; padding:.38rem .75rem; font-size:.78rem; font-weight:600; }
    .section-label { color:#568068; font-weight:700; font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; margin:.35rem 0 .15rem; }
    .soft-card { border:1px solid #e1ece2; border-radius:18px; background:#fff; padding:1.1rem 1.25rem; box-shadow:0 7px 22px #18332808; }
    .result-card { padding:1.45rem 1.6rem; border:1px solid #cce5d1; border-radius:20px; background:linear-gradient(135deg,#f1faf2,#fff); box-shadow:0 10px 30px #1b653010; }
    .result-kicker { color:#568068; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; }
    .result-crop { color:#145a36; font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; margin:.25rem 0; }
    .stButton > button, .stFormSubmitButton > button { border-radius:12px; border:0; min-height:3rem; font-weight:700; letter-spacing:.025em; transition:all .18s ease; }
    .stFormSubmitButton > button[kind="primary"], .stButton > button[kind="primary"] { background:linear-gradient(110deg,#17683f,#278650); color:white; box-shadow:0 7px 17px #17683f30; }
    .stFormSubmitButton > button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover { background:#105332; transform:translateY(-1px); }
    [data-testid="stMetric"] { background:#fff; border:1px solid #e2ece3; border-radius:15px; padding:.8rem 1rem; box-shadow:0 5px 18px #18332808; }
    [data-testid="stMetricLabel"] { color:#465d4e; }
    [data-testid="stMetricValue"] { color:#183328; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius:10px; background:#fff; border-color:#bdcfc1; }
    div[data-baseweb="input"] input, div[data-baseweb="select"] input { color:#183328; }
    [data-testid="stImage"] { background:#fff; border:1px solid #e1ece2; border-radius:14px; padding:.6rem; box-shadow:0 5px 18px #18332808; }
    .footer { margin-top:2.5rem; padding:1rem .2rem .25rem; border-top:1px solid #deeadf; color:#53665b; font-size:.78rem; display:flex; flex-wrap:wrap; justify-content:space-between; gap:.5rem; }
    @media (max-width:700px) { [data-testid="stMainBlockContainer"] { padding:1rem; border-radius:16px; }.hero {padding:1.5rem;border-radius:18px}.hero:after{right:-2%;top:22%;font-size:5rem}.result-crop{font-size:1.7rem} }
    </style>""".replace("__FIELD_IMAGE__", FIELD_IMAGE).replace("__FARMERS_IMAGE__", FARMERS_IMAGE),
    unsafe_allow_html=True,
)


def get_results():
    """Read saved evaluation results without retraining or hardcoding metrics."""
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def render_hero():
    st.markdown(
        """<div class="hero">
          <div class="eyebrow">AI-powered agriculture</div>
          <h1>CROPWISE</h1>
          <p>Find a data-informed crop match from your soil nutrients and growing conditions.
          Simple inputs, a trained KNN model, and a clearer place to start.</p>
          <div class="hero-pills"><span class="pill">🌿 Crop Recommendation System</span>
          <span class="pill">◉ KNN Machine Learning</span></div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        "<div class='footer'><span><b>CropWise</b> · AI-Powered Agriculture · Academic Mini Project</span>"
        "<span>For educational use. Confirm crop choices with local agricultural expertise.</span></div>",
        unsafe_allow_html=True,
    )


results = get_results()
st.sidebar.markdown("<div class='brand-mark'><span class='brand-icon'>🌱</span>CROPWISE</div>", unsafe_allow_html=True)
st.sidebar.caption("AI-powered crop recommendation")
page = st.sidebar.radio("Explore", ["Home", "Crop Prediction", "Model Performance", "About"], label_visibility="collapsed")
st.sidebar.divider()
if MODEL_PATH.exists():
    st.sidebar.success("KNN model ready")
else:
    st.sidebar.warning("Model not found")
st.sidebar.caption("Soil insights for smarter growing")

render_hero()

if page == "Home":
    st.markdown("<div class='section-label'>Your growing conditions, understood</div>", unsafe_allow_html=True)
    st.header("A thoughtful starting point for crop selection")
    st.write("CropWise brings soil nutrients and environmental measurements together to suggest a crop label learned from the project dataset.")
    overview = st.columns(3)
    for column, icon, title, description in [
        (overview[0], "🧪", "Soil nutrients", "Include nitrogen, phosphorus, and potassium levels."),
        (overview[1], "☀️", "Growing conditions", "Add temperature, humidity, pH, and rainfall."),
        (overview[2], "🤖", "KNN recommendation", "The saved, scaled model compares your inputs with labeled examples."),
    ]:
        with column:
            st.markdown(f"<div class='soft-card'><div style='font-size:1.5rem'>{icon}</div><h3 style='font-size:1.05rem;margin:.55rem 0 .25rem'>{title}</h3><div style='color:#65796e'>{description}</div></div>", unsafe_allow_html=True)
    st.write("")
    if results.get("status") == "evaluated":
        st.subheader("Model snapshot")
        stats = st.columns(4)
        stats[0].metric("Dataset records", f"{results.get('records_after_deduplication', '—'):,}" if isinstance(results.get("records_after_deduplication"), int) else "—")
        stats[1].metric("Crop classes", results.get("class_count", "—"))
        stats[2].metric("Selected K", results.get("selected_k", "—"))
        accuracy = results.get("testing_accuracy")
        stats[3].metric("Test accuracy", f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else "—")
    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.subheader("How it works")
        st.write("Enter the seven measurements on the prediction page. The existing preprocessing and saved KNN pipeline process them in the trained feature order and return one crop class.")
    with right:
        st.subheader("Get started")
        st.write("Use values measured for your field. CropWise is an educational project and should complement local agronomic guidance.")
        if not MODEL_PATH.exists():
            st.warning("The trained model is not available. See README.md for dataset and training steps.")

elif page == "Crop Prediction":
    st.markdown("<div class='section-label'>Field input dashboard</div>", unsafe_allow_html=True)
    st.header("Tell us about your field")
    st.write("Enter the soil and climate values below. Use the units and ranges shown for each measurement.")
    with st.form("prediction_form"):
        st.markdown("**01 · Soil nutrients**")
        nutrient_cols = st.columns(3)
        with nutrient_cols[0]:
            nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0, step=1.0)
        with nutrient_cols[1]:
            phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=50.0, step=1.0)
        with nutrient_cols[2]:
            potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=50.0, step=1.0)

        st.markdown("**02 · Climate conditions**")
        climate_cols = st.columns(2)
        with climate_cols[0]:
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1)
        with climate_cols[1]:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.1)

        st.markdown("**03 · Soil and water**")
        ground_cols = st.columns(2)
        with ground_cols[0]:
            soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        with ground_cols[1]:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

        submitted = st.form_submit_button("🌿  RECOMMEND MY CROP", type="primary", use_container_width=True)

    if submitted:
        # Keep the original FEATURES order expected by predict_crop and the saved pipeline.
        input_values = [nitrogen, phosphorus, potassium, temperature, humidity, soil_ph, rainfall]
        if len(input_values) != len(FEATURES):
            st.error("Please provide all seven required measurements.")
        elif not MODEL_PATH.exists():
            st.error("No trained model found. Train the model with the real dataset first.")
        else:
            try:
                # Use the existing prediction function; do not create a second model path.
                crop = predict_crop(input_values)
                st.session_state["cropwise_prediction"] = {"crop": crop, "values": input_values}
            except (ValueError, FileNotFoundError, OSError) as error:
                st.error(f"Could not make a prediction: {error}")
                st.session_state.pop("cropwise_prediction", None)

    prediction = st.session_state.get("cropwise_prediction")
    if prediction:
        st.write("")
        st.markdown(
            f"<div class='result-card'><div class='result-kicker'>Your KNN recommendation</div>"
            f"<div class='result-crop'>🌾 {prediction['crop'].title()}</div>"
            "<div style='color:#65796e'>Predicted from the seven field measurements you entered.</div></div>",
            unsafe_allow_html=True,
        )
        with st.expander("Review the values used"):
            st.dataframe({"Input feature": FEATURES, "Your value": prediction["values"]}, hide_index=True, use_container_width=True)
        st.caption("This is a model classification, not a yield guarantee or a substitute for local agronomic advice.")

elif page == "Model Performance":
    st.markdown("<div class='section-label'>Evidence from the held-out data</div>", unsafe_allow_html=True)
    st.header("Model performance")
    if results.get("status") != "evaluated":
        st.info("Actual evaluation results are not available yet. Run the training and evaluation commands first.")
        if results.get("cv_accuracy_by_k"):
            st.write("Cross-validation accuracy by K", results["cv_accuracy_by_k"])
    else:
        metrics = [("Training accuracy", "training_accuracy"), ("Test accuracy", "testing_accuracy"),
                   ("Weighted precision", "weighted_precision"), ("Weighted recall", "weighted_recall"),
                   ("Weighted F1", "weighted_f1_score")]
        columns = st.columns(len(metrics))
        for column, (label, key) in zip(columns, metrics):
            value = results.get(key)
            column.metric(label, f"{value:.1%}" if isinstance(value, (int, float)) else "—")
        st.caption(f"K = {results.get('selected_k', '—')} · {results.get('test_records', '—')} held-out test records · Euclidean distance")
        for filename, title in [("confusion_matrix.png", "Confusion matrix"),
                                ("k_value_comparison.png", "K selection by cross-validation"),
                                ("crop_distribution.png", "Crop class distribution")]:
            figure_path = ROOT / "reports" / "figures" / filename
            if figure_path.exists():
                st.subheader(title)
                st.image(str(figure_path), use_container_width=True)
        with st.expander("View classification report"):
            st.json(results.get("classification_report", {}))

elif page == "About":
    st.markdown("<div class='section-label'>The method behind the recommendation</div>", unsafe_allow_html=True)
    st.header("About CropWise")
    st.subheader("Supervised learning")
    st.write("A supervised classifier learns from examples that pair input measurements with known crop labels. Here, seven soil and climate features are used to predict one of the crop labels represented in the dataset.")
    st.subheader("K-Nearest Neighbors")
    st.write("KNN selects the majority label among nearby training examples. This project compares K = 3, 5, 7, 9, and 11 using stratified five-fold cross-validation on training data, with Euclidean distance after feature scaling.")
    st.subheader("Dataset features")
    st.write("The Crop Recommendation Dataset includes the following model inputs:")
    st.dataframe({"Feature": FEATURES, "Description": ["Nitrogen content", "Phosphorus content", "Potassium content", "Temperature in °C", "Relative humidity in %", "Soil pH", "Rainfall in mm"]}, hide_index=True, use_container_width=True)
    st.subheader("Limitations")
    st.write("Recommendations are limited to crops and conditions in the training dataset. The model does not account for every local cultivar, soil sampling uncertainty, irrigation plan, pest pressure, market conditions, or future weather. Confirm decisions with local agricultural expertise.")

render_footer()
