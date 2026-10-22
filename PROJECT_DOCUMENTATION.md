# Project Documentation

## Locked project title
PRIVACY-PRESERVING FEDERATED LEARNING FRAMEWORK FOR
MULTI-HOSPITAL MEDICAL DATA ANALYSIS

## Case study
Chronic Kidney Disease (CKD) prediction.

## Research workflow
1. Prepare and preprocess datasets for four simulated clients.
2. Use a common set of 19 input features.
3. Train and aggregate client updates using Federated Averaging.
4. Compare Normal FL, Fixed-DP and Proposed Adaptive-DP.
5. Present research results and privacy-attack evaluation.
6. Demonstrate a dashboard prediction and downloadable PDF report.

## API smoke tests
The project owner reported successful tests for:
- GET /api/health: HTTP 200, model loaded.
- POST /api/predict: HTTP 200 with a 19-feature response.
- POST /api/patient-report: HTTP 200 with a PDF response.

These are reported software tests, not independent clinical validation.
The browser-based end-to-end workflow should also be tested.

## Safety and interpretation
The prototype is for academic research demonstration only. Its predicted
probability is a model output, not a clinically calibrated patient risk.
Verify that feature encodings and preprocessing match the original
training pipeline before interpreting any prediction.
