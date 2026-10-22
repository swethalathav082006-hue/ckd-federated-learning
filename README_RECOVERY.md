# CKD Federated Learning — Recovery Source Pack

Locked title:
PRIVACY-PRESERVING FEDERATED LEARNING FRAMEWORK FOR MULTI-HOSPITAL MEDICAL DATA ANALYSIS

Case study: Chronic Kidney Disease (CKD) prediction.

## What was recovered
- `app.py`: Flask backend source recovered from the uploaded notebook's embedded code.
- `index.html`, `style.css`, `script.js`: dashboard frontend source recovered from the notebook.
- `requirements.txt`: basic Python dependencies.
- `CKD_Project_Recovery.ipynb`: uploaded notebook retained as the full reference for data preparation, training, evaluation, and later project edits.

## Important limitation
This is a **source-recovery pack**, not yet a fully runnable final prototype. The uploaded notebook does not contain the trained `.pth` model files or fitted `.joblib` preprocessing artifacts as attached files. Those artifacts must be recreated by rerunning the appropriate dataset preparation/training/export steps in a connected Colab runtime. The dashboard's prediction endpoint will report that the model is unavailable until the model and preprocessing files are present.

Folder locations expected by `app.py`:
- `models/adaptive_dp_fl_model.pth`
- `preprocessors/hospital_1_imputer.joblib` and `hospital_1_scaler.joblib`
- same imputer/scaler pair for hospitals 2, 3, and 4.

## Run after model artifacts are recreated
1. In a fresh Colab runtime, upload/extract this pack.
2. Install dependencies: `pip install -r requirements.txt reportlab`
3. Ensure the model and preprocessor files exist at the paths above.
4. Start with `python app.py` (default port 5000).
5. Open `/api/health` to verify model-loaded status before using predictions.

## Research caution
This is a research prototype using four simulated clients derived from public datasets. Outputs are not medical diagnoses. Privacy claims should not be described as formally differentially private without a documented privacy accountant and validated implementation.
