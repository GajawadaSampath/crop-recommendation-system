"""Build an academic PDF report, using explicit pending markers until evaluation exists."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "model_results.json"
OUTPUT = ROOT / "reports" / "Crop_Recommendation_System_Report.pdf"


def generate_report():
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    evaluated = results.get("status") == "evaluated"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
    name="Caption",
    parent=styles["Normal"],
    fontSize=9,
    leading=12,
    alignment=1,
    spaceAfter=10
))
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=colors.HexColor("#245b35"), fontSize=24, leading=30))
    story = [Spacer(1, 4*cm), Paragraph("Crop Recommendation System", styles["CoverTitle"]),
             Paragraph("CSE_16 · Academic Machine Learning Mini Project", styles["Heading2"]),
             Spacer(1, 1.3*cm), Paragraph("Student names: ______________________________", styles["BodyText"]),
             Paragraph("Institution: __________________________________", styles["BodyText"]),
             Paragraph("Faculty guide: _______________________________", styles["BodyText"]), Spacer(1, 3*cm),
             Paragraph("Report status: " + ("Experimental evaluation completed" if evaluated else "Template — dataset and experimental results pending"), styles["Italic"]), PageBreak()]
    sections = [
        ("Abstract", "This project presents a supervised multi-class crop recommendation system. A K-Nearest Neighbors classifier uses nitrogen, phosphorus, potassium, temperature, humidity, soil pH, and rainfall. The dataset is split into stratified training and testing partitions; feature scaling and K selection are performed without fitting on the held-out test partition."),
        ("1. Introduction", "Agricultural crop selection depends on soil nutrients and environmental conditions. This mini-project demonstrates a reproducible classification workflow and a Streamlit interface for exploring the resulting model."),
        ("2. Problem Statement", "Provide a crop class recommendation from seven numeric inputs using a trained model, while reporting genuine held-out evaluation results."),
        ("3. Objectives", "Load and validate the real dataset; compare five K values with cross-validation; fit a scaled KNN pipeline; evaluate on a held-out test set; and provide an accessible dashboard."),
        ("4. Existing and Proposed Systems", "Manual selection depends on experience and local references. The proposed educational system applies a supervised KNN classifier to a labeled dataset and exposes the prediction through a web interface."),
        ("5. Dataset Description", "Source: Atharva Ingle, Crop Recommendation Dataset on Kaggle. Expected CSV: dataset/Crop_recommendation.csv. Expected fields: N, P, K, temperature, humidity, ph, rainfall, label. No synthetic records are used."),
        ("6. Feature Descriptions", "N, P, K represent soil macronutrient measurements; temperature is in °C; humidity is relative humidity; ph represents soil acidity/alkalinity; rainfall is precipitation; label is the crop class."),
        ("7. Supervised Learning and KNN", "Supervised learning estimates a mapping from labeled input examples to a target class. KNN predicts the majority label among the K nearest examples. Here Euclidean distance is applied after StandardScaler normalization, with K selected from 3, 5, 7, 9, and 11."),
        ("8. System Architecture", "CSV → validation and duplicate handling → stratified train/test split → cross-validated Pipeline (StandardScaler + KNN) → saved model → held-out evaluation and Streamlit prediction interface."),
        ("9. Methodology and Preprocessing", "Pandas checks required columns, missing values, duplicates, numeric feature types, finite values, and target labels. Duplicate rows are removed. Data is split 80:20 with stratification and fixed random seed 42. The scaler is fitted inside each CV fold or on training data only."),
        ("10. Model Implementation", "The project uses scikit-learn KNeighborsClassifier with Euclidean distance in a Pipeline. Five-fold stratified cross-validation compares candidate K values using training data. The selected pipeline is fitted on the entire training partition and serialized with Joblib."),
        ("11. Training, Testing, and Metrics", "Training accuracy and test accuracy are calculated from the corresponding partitions. Weighted precision, recall, F1-score, a per-class classification report, and confusion matrix are generated on the held-out test set."),
        ("12. Experimental Results", ""),
        ("13. Figures and Application Screenshots", "Evaluation figures are produced after training. Application screenshots should be captured from a real running session and stored in screenshots/."),
        ("14. Advantages", "The workflow is reproducible, uses a leakage-safe pipeline, compares K values on training data, and presents clear input fields and evaluation outputs."),
        ("15. Limitations", "The model is limited by its dataset's geography, class coverage, measurement quality, and feature set. It does not account for local varieties, irrigation, pests, economics, or future weather, and is not professional agronomic advice."),
        ("16. Future Enhancements", "Validate with region-specific data, document collection protocols, add model monitoring and versioning, and enrich agronomic context while retaining KNN as this project's required primary method."),
        ("17. Conclusion", "The project provides an end-to-end KNN classification implementation. Conclusions about predictive performance must be based on the real dataset's held-out test evaluation."),
        ("18. References", "1. Atharva Ingle, Crop Recommendation Dataset, Kaggle: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n2. Scikit-learn documentation: https://scikit-learn.org/stable/\n3. Streamlit documentation: https://docs.streamlit.io/")]
    for heading, body in sections:
        story.append(Paragraph(heading, styles["Heading1"]))
        if body: story.append(Paragraph(body.replace("\n", "<br/>"), styles["BodyText"]))
        if heading == "12. Experimental Results":
            if evaluated:
                rows=[["Metric","Observed value"], ["Selected K",str(results["selected_k"])],
                      ["Training accuracy",f"{results['training_accuracy']:.4f}"],
                      ["Testing accuracy",f"{results['testing_accuracy']:.4f}"],
                      ["Weighted precision",f"{results['weighted_precision']:.4f}"],
                      ["Weighted recall",f"{results['weighted_recall']:.4f}"],
                      ["Weighted F1",f"{results['weighted_f1_score']:.4f}"]]
                table=Table(rows, colWidths=[7*cm,7*cm]); table.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dceedd")),("GRID",(0,0),(-1,-1),.5,colors.grey),
                    ("PADDING",(0,0),(-1,-1),7)])); story.append(table)
            else: story.append(Paragraph("PENDING — place the real CSV in dataset/Crop_recommendation.csv, run training and evaluation, then regenerate this report. No performance values have been fabricated.", styles["BodyText"]))
            story.append(Spacer(1, 0.4*cm))
            for name, caption in [("confusion_matrix.png","Figure 1. Confusion matrix on the held-out test set."),
                                  ("k_value_comparison.png","Figure 2. Mean cross-validation accuracy by K."),
                                  ("crop_distribution.png","Figure 3. Class distribution in the dataset.")]:
                path=ROOT / "reports" / "figures" / name
                if evaluated and path.exists(): story.extend([Image(str(path), width=14*cm, height=9*cm), Paragraph(caption, styles["Caption"])])
                else: story.append(Paragraph("Pending figure: " + caption, styles["Italic"]))
        elif heading == "13. Figures and Application Screenshots":
            shots=list((ROOT / "screenshots").glob("*.png"))
            if shots:
                story.extend([Image(str(shots[0]), width=15*cm, height=8.5*cm), Paragraph("Figure 4. Application screenshot.", styles["Caption"])])
            else: story.append(Paragraph("Pending — capture screenshots after launching the Streamlit application.", styles["Italic"]))
        story.append(Spacer(1, 0.3*cm))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(A4[0]/2, 1.2*cm, f"Crop Recommendation System · Page {doc.page}")
        canvas.restoreState()
    SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                      topMargin=1.8*cm, bottomMargin=2*cm, title="Crop Recommendation System Report").build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Report saved: {OUTPUT}")


if __name__ == "__main__":
    generate_report()
