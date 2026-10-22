# Privacy-Preserving Federated Learning Framework
## Multi-Hospital Medical Data Analysis — CKD Case Study

### Project scope
This is a research demonstration of federated learning for CKD
classification using four simulated hospital clients and public datasets.
It does not connect to real hospitals and is not a clinical diagnostic tool.

### Main components
- app.py: Flask backend, prediction API and PDF report endpoint
- index.html: Dashboard interface
- style.css: Dashboard styling
- script.js: Frontend interactions and API requests
- models/: Saved model weights, if present
- preprocessors/: Hospital-specific preprocessing objects, if present
- graphs/: Research graphs, if present
- results/: Experimental results, if present
- requirements.txt: Python dependencies

### Model and experiment
The common feature set contains 19 features.
The research compares Normal FL, Fixed-DP and Proposed Adaptive-DP.
The saved prediction endpoint uses the Adaptive-DP model when loaded.

### Run locally
1. Install Python 3.10 or a compatible version.
2. Install dependencies:
   pip install -r requirements.txt
3. Start the application:
   python app.py
4. Open the local address printed by Flask.

If port 5000 is already occupied, change the app's run port or stop
the conflicting process before starting another server.

### API endpoints
- GET /api/health
- POST /api/predict
- POST /api/patient-report

Prediction requests require the hospital selection and the expected
19 model features. Refer to app.py for the exact request schema.

### Important limitations
- Hospital clients are simulated; they are not real hospitals.
- Public datasets may differ in encoding and data collection.
- Preprocessing must match the original model training pipeline.
- A successful API response does not establish clinical validity.
- The noise mapping is an experimental research design, not a proven
  clinical privacy guarantee.
- Do not use this prototype for real patient diagnosis or treatment.
