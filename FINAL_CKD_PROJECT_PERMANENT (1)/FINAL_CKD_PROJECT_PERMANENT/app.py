import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

TITLE = "PRIVACY-PRESERVING FEDERATED LEARNING FRAMEWORK FOR MULTI-HOSPITAL MEDICAL DATA ANALYSIS"
CASE_STUDY = "Chronic Kidney Disease (CKD)"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CKD_Federated_Project_Output"
MODEL_DIR = OUT / "model"
RESULT_DIR = OUT / "results"
GRAPH_DIR = OUT / "graphs"
REPORT_DIR = OUT / "reports"

FEATURES = ["age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc","appet","pe","ane","htn","dm","cad"]
NUMERIC = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]
CATEGORICAL = ["rbc","pc","pcc","ba","appet","pe","ane","htn","dm","cad"]
CAT_MAP = {"rbc":{"normal":0.0,"abnormal":1.0},"pc":{"normal":0.0,"abnormal":1.0},"pcc":{"notpresent":0.0,"present":1.0},"ba":{"notpresent":0.0,"present":1.0},"appet":{"good":0.0,"poor":1.0},"pe":{"no":0.0,"yes":1.0},"ane":{"no":0.0,"yes":1.0},"htn":{"no":0.0,"yes":1.0},"dm":{"no":0.0,"yes":1.0},"cad":{"no":0.0,"yes":1.0}}

class CKDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(24,64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x):
        return self.network(x).squeeze(1)

def load_artifacts():
    config_path = MODEL_DIR / "config.json"
    state_path = MODEL_DIR / "Adaptive_DP_Global_Model.pt"
    if not config_path.exists() or not state_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run train_colab.py first, then copy CKD_Federated_Project_Output into this Space.")
    config = json.loads(config_path.read_text())
    state = torch.load(state_path, map_location="cpu")
    model = CKDModel()
    model.load_state_dict(state)
    model.eval()
    return config, model

CONFIG, MODEL = load_artifacts()
MEDIANS = CONFIG.get("medians", {})
RANGES = CONFIG.get("ranges", {})


def norm(v):
    return re.sub(r"[^a-z0-9]+", "", str(v).strip().lower())

def encode_row(values):
    row = dict(zip(FEATURES, values))
    out = []
    for f in FEATURES:
        v = row.get(f)
        if f in NUMERIC:
            try:
                z = float(v)
            except Exception:
                z = float(MEDIANS.get(f, 0.0))
            lo, hi = RANGES.get(f, (0.0, 1.0))
            z = min(max(z, lo), hi)
            # Fixed clinical-range normalization must match training.
            z = 2.0 * (z - lo) / (hi - lo) - 1.0 if hi > lo else 0.0
            out.append(z)
        else:
            c = norm(v)
            out.append(float(CAT_MAP.get(f, {}).get(c, 0.0)))
    return np.asarray([out], dtype=np.float32)


def predict_values(*values):
    X = encode_row(values)
    with torch.no_grad():
        p = float(torch.sigmoid(MODEL(torch.tensor(X))).item())
    label = "CKD" if p >= 0.5 else "Not CKD"
    return label, f"{p*100:.2f}%"


def make_prediction_pdf(*values):
    label, probability = predict_values(*values)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "latest_prediction_report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph(TITLE, styles["Title"]), Spacer(1,10), Paragraph("CKD Case Study — Interactive Prediction Report", styles["Heading2"]), Spacer(1,8)]
    story.append(Paragraph("This result is an academic prototype output and is not a medical diagnosis.", styles["BodyText"]))
    story.append(Spacer(1,12))
    rows = [["Feature","Value"]] + [[f.replace("_"," ").title(), str(v)] for f,v in zip(FEATURES, values)]
    rows += [["Prediction", label], ["CKD Probability", probability]]
    table = Table(rows, colWidths=[210,260])
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
    story.append(table)
    doc.build(story)
    return str(path)


def project_report_path():
    p = REPORT_DIR / "project_report.pdf"
    return str(p) if p.exists() else None

def graph(name):
    p = GRAPH_DIR / name
    return str(p) if p.exists() else None

def load_csv(name):
    p = RESULT_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame({"Status":[f"{name} not found — run training first."]})

FINAL = load_csv("final_metrics.csv")
HOSPITAL = load_csv("hospital_results.csv")
NOISE = load_csv("noise_table.csv")
MIA = load_csv("mia_results.csv")
NORMAL = load_csv("normal_history.csv")
FIXED = load_csv("fixed_dp_history.csv")
ADAPTIVE = load_csv("adaptive_dp_history.csv")

inputs = [
    gr.Number(value=50,label="Age"), gr.Number(value=80,label="Blood Pressure"), gr.Number(value=1.020,label="Urine Specific Gravity"), gr.Number(value=1,label="Urine Albumin"),
    gr.Number(value=0,label="Urine Sugar"), gr.Dropdown(["normal","abnormal"],value="normal",label="Red Blood Cells"), gr.Dropdown(["normal","abnormal"],value="normal",label="Pus Cells"), gr.Dropdown(["notpresent","present"],value="notpresent",label="Pus Cell Clumps"),
    gr.Dropdown(["notpresent","present"],value="notpresent",label="Bacteria"), gr.Number(value=100,label="Random Blood Glucose"), gr.Number(value=40,label="Blood Urea"), gr.Number(value=1.2,label="Serum Creatinine"),
    gr.Number(value=140,label="Sodium"), gr.Number(value=4.5,label="Potassium"), gr.Number(value=13,label="Hemoglobin"), gr.Number(value=40,label="Packed Cell Volume"),
    gr.Number(value=8000,label="White Blood Cell Count"), gr.Number(value=5,label="Red Blood Cell Count"), gr.Dropdown(["good","poor"],value="good",label="Appetite"), gr.Dropdown(["no","yes"],value="no",label="Pedal Edema"),
    gr.Dropdown(["no","yes"],value="no",label="Anemia"), gr.Dropdown(["no","yes"],value="no",label="Hypertension"), gr.Dropdown(["no","yes"],value="no",label="Diabetes Mellitus"), gr.Dropdown(["no","yes"],value="no",label="Coronary Artery Disease")
]

with gr.Blocks(title="Privacy-Preserving Federated Learning — CKD") as demo:
    gr.Markdown(f"# {TITLE}\n### {CASE_STUDY}")
    with gr.Tab("Project Summary"):
        gr.Markdown("""### Final Prototype\n\nThis dashboard presents the trained global model, simulated multi-hospital federated experiments, hospital-wise missing-feature settings, adaptive privacy perturbation, and a confidence-based membership-inference attack evaluation.\n\n**Important:** Hospitals are simulated clients created from public datasets. The missingness-to-noise mapping is a proposed experimental mechanism, not a clinically validated rule. The privacy perturbation is a research prototype and does not by itself establish a formal (ε,δ)-DP guarantee.""")
        gr.Dataframe(FINAL.round(4), interactive=False, label="Final Performance")
        gr.Dataframe(MIA.round(4), interactive=False, label="MIA Evaluation")
        gr.File(value=project_report_path(), label="Complete Project Report PDF")
    with gr.Tab("Hospital Analysis"):
        gr.Dataframe(HOSPITAL.round(4), interactive=False, label="Hospital-Wise Results")
        gr.Dataframe(NOISE, interactive=False, label="Missingness / Adaptive Noise")
        gr.Image(value=graph("missingness_vs_noise.png"), type="filepath", label="Missingness vs Adaptive Noise")
    with gr.Tab("Federated Training"):
        gr.Image(value=graph("accuracy_vs_round.png"), type="filepath", label="Accuracy Across Rounds")
        gr.Image(value=graph("f1_vs_round.png"), type="filepath", label="F1 Across Rounds")
        gr.Image(value=graph("final_metric_comparison.png"), type="filepath", label="Final Metric Comparison")
        gr.Dataframe(NORMAL.round(4), interactive=False, label="Normal FL")
        gr.Dataframe(FIXED.round(4), interactive=False, label="Fixed-DP FL")
        gr.Dataframe(ADAPTIVE.round(4), interactive=False, label="Proposed Adaptive-DP FL")
    with gr.Tab("Privacy Evaluation"):
        gr.Image(value=graph("mia_comparison.png"), type="filepath", label="Membership Inference Attack Comparison")
        gr.Image(value=graph("cm_adaptive_dp_fl.png"), type="filepath", label="Adaptive-DP Confusion Matrix")
        gr.Markdown("### Proposed adaptive mapping\n- 5% missing → 0.05 noise\n- 15% missing → 0.10 noise\n- 25% missing → 0.20 noise\n- 40% missing → 0.30 noise\n\nThese values are experimental settings selected for this project.")
    with gr.Tab("CKD Prediction"):
        gr.Markdown("Enter clinical values for an academic prototype prediction. **This is not a medical diagnosis.**")
        for i in range(0,len(inputs),4):
            with gr.Row():
                for w in inputs[i:i+4]: w.render()
        with gr.Row():
            predict_btn = gr.Button("Predict CKD")
            pdf_btn = gr.Button("Generate Prediction PDF")
        pred = gr.Textbox(label="Prediction")
        prob = gr.Textbox(label="CKD Probability")
        pred_pdf = gr.File(label="Prediction Report PDF")
        predict_btn.click(predict_values, inputs=inputs, outputs=[pred,prob])
        pdf_btn.click(make_prediction_pdf, inputs=inputs, outputs=pred_pdf)
    with gr.Tab("Downloads"):
        gr.Markdown("### Project files\nThe complete report and generated result files are stored with the Space repository after deployment.")
        gr.File(value=project_report_path(), label="Project Report")
        for fn in ["final_metrics.csv","hospital_results.csv","noise_table.csv","mia_results.csv","normal_history.csv","fixed_dp_history.csv","adaptive_dp_history.csv"]:
            p=RESULT_DIR/fn
            if p.exists(): gr.File(value=str(p), label=fn)

if __name__ == "__main__":
    demo.launch()
