# Integration instructions

## 1. Add `src/monitoring.py`
Copy this file into your repository at `src/monitoring.py`.

## 2. Add a monitoring section to `app/streamlit_app.py`
Near the top of your existing Streamlit app, add:

```python
from src.monitoring import render_ecg_monitoring
```

Then add a section where you want the uploader to appear. If your app is already using tabs, add a tab for monitoring; otherwise, add this near the end of the app:

```python
st.divider()
render_ecg_monitoring("models/random_forest.joblib")
```

If your saved Random Forest model has a different path, update the path accordingly.

## 3. Dependencies
Ensure `requirements.txt` includes:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
wfdb
streamlit
joblib
```

Install any missing packages inside your virtual environment:

```bash
pip install -r requirements.txt
```

## 4. Run locally

```bash
streamlit run app/streamlit_app.py
```

## Important compatibility caveats
- This code expects `models/random_forest.joblib` to be the fitted Random Forest trained on the **252 raw waveform samples**, not the feature-based RF trained on 9 statistical features.
- The training pipeline used 252 samples at 360 Hz with the annotation location as the window reference. The upload pipeline uses a simple signal-processing detector to estimate beat locations. Those estimates can be misaligned.
- CSV amplitude units and lead configuration should resemble the training signal. Different device units, leads, filtering, or gain may materially change predictions.
- The simple peak detector is only a demonstration detector, not a validated QRS detector.
- The repeated-V rule is configurable and purely illustrative; it is not a medical threshold.
- For WFDB upload, select the matching `.hea` and `.dat` files together. The first lead is used.
