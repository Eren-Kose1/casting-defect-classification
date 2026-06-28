# Explainable CNN-Based Casting Product Defect Classification

Final project for the Machine Learning and Smart Systems module by **Eren Köse**.
It classifies casting product images as **defective** or **acceptable (OK)**, compares a custom CNN
against two transfer-learning models, explains the predictions with Grad-CAM and SHAP, tests
robustness on degraded images, and ships a small web app for inspection.

## Results (test set: 715 images)

| Model | Accuracy | Precision | Defective recall | F1 | Missed defects | Params | Inference |
|---|---|---|---|---|---|---|---|
| **Custom CNN** | **99.30%** | 99.12% | **99.78%** | **99.45%** | **1** | **0.26 M** | **2.6 ms** |
| EfficientNetV2-B0 | 99.02% | 99.78% | 98.68% | 99.22% | 6 | 5.92 M | 17.5 ms |
| ConvNeXt-Tiny | 98.46% | 98.25% | 99.34% | 98.79% | 3 | 27.8 M | 13.7 ms |

The lightweight custom CNN is the best overall choice: highest accuracy, fewest missed defects,
about 100x smaller and 7x faster than the pretrained models. Transfer learning did not improve
results on this dataset.

**Robustness:** the custom CNN tolerates mild blur but is sensitive to noise and darkening, where it
becomes over-cautious and flags most parts as defective. This is a fail-safe direction but shows the
model needs more varied training data before real deployment.

## Repository structure
- `Casting_Defect_Classification.ipynb` - the executed Kaggle notebook (full pipeline)
- `app/` - Streamlit web front-end (loads the custom CNN, shows Grad-CAM)
- `results/` - figures, metric tables, and the robustness results
- `Proposal_Casting_Product_Defect_Classification.md` - the project proposal

## Dataset
Casting Product Image Data for Quality Inspection (Kaggle, by Ravirajsinh Dabhi):
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

## Links
- Kaggle notebook: https://www.kaggle.com/code/erenkose1/casting-defect-classification
- Live web app: https://casting-defect-classification-cuvelj7hse4b9erntiq45y.streamlit.app
- Report (Overleaf PDF): _to be added_

## Run the web app locally
```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```
