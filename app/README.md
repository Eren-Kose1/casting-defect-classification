# Casting Defect Inspector (web front-end)

A small Streamlit app for the project *Explainable CNN-Based Casting Product Defect Classification*.
It loads the trained custom CNN, predicts whether an uploaded casting image is **defective** or **OK**,
and shows a **Grad-CAM** heatmap explaining the decision.

## Files
- `streamlit_app.py` - the app
- `best_model.keras` - the trained custom CNN exported from the Kaggle notebook
- `requirements.txt` - dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open the local URL it prints (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud (free)
1. Push this repository to a public GitHub repo.
2. Go to share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick the repo, and set the main file to `app/streamlit_app.py`.
4. Deploy. Streamlit installs `requirements.txt` and gives you a public URL.

The model is the lightweight custom CNN (about 0.26 M parameters), so it loads quickly and runs on CPU.
