# Explainable CNN-Based Casting Product Defect Classification for Industrial Quality Inspection

**Custom CNN vs. EfficientNetV2 vs. ConvNeXtTiny with Grad-CAM, SHAP, and Web Deployment**

*Field 1: Industrial Inspection and Manufacturing — Topic 2: Casting Product Defect Classification*

Author: **Eren Köse** — Matriculation No. 62675606
Department of Business, University of Europe for Applied Sciences, Potsdam, Germany

> This document is the working proposal/spec for the project. It consolidates the professor's Idea Book entry for this topic (the authoritative source for the Research Questions) with concrete dataset details, so it can anchor the Kaggle notebook and the Overleaf report.

---

## Project Description

This project develops an explainable image-classification system for detecting **defective** and **non-defective (OK)** casting products. Casting defects such as blowholes, pinholes, burrs, shrinkage defects, surface deformation, and mould-material defects are common in metal manufacturing and reduce product reliability. In industrial settings, defective casting products must be identified before assembly, distribution, or further machining. Manual inspection is slow and inconsistent, especially when surface defects are small or visually ambiguous.

The project compares a **custom CNN** baseline with **EfficientNetV2** and **ConvNeXtTiny** for **binary** classification of casting images into **defective** and **non-defective/OK** categories. The work is positioned as a practical machine-vision inspection system combining performance evaluation, explainability, and deployment. The contribution emphasises **industrial applicability**, **false-negative reduction**, and **visual validation** of model decisions using Explainable AI (XAI). A false negative (a defective product classified as acceptable) is the most costly error, so defective-class recall is the priority metric.

## Two Sample Applications

1. **Casting-line quality inspection:** A prototype that screens casting product images and flags defective parts before packaging or machining.
2. **Manufacturing decision-support dashboard:** A web tool that lets operators review the predicted defect status (defective/OK) with a confidence score and inspect a Grad-CAM explanation heatmap.

## Methodology

Images are organised into **defective** and **OK** folders, resized to **224 × 224**, converted from grayscale to three channels (for the pre-trained backbones), normalised, and augmented using rotation, brightness variation, contrast adjustment, zoom, and horizontal flipping. Rotation is appropriate because casting parts may appear in different orientations. The dataset is checked for class imbalance, since real-world defect datasets contain unequal defective and OK samples; **class weights** are used if needed.

The custom CNN is the from-scratch baseline. EfficientNetV2 and ConvNeXtTiny are trained with **transfer learning** using a binary classification head and a two-stage strategy (frozen-backbone feature extraction → partial fine-tuning), with early stopping, learning-rate scheduling, and model checkpointing. Evaluation includes accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, **false-negative rate**, training time, inference time, model size, and parameter count, with **defective-class recall** emphasised throughout.

**Grad-CAM** verifies whether the models attend to holes, burrs, cracks, missing regions, or surface irregularities. **SHAP** confirms whether those same regions contribute positively to the defective prediction. The selected model is deployed as a **Streamlit** or **Gradio / Hugging Face Spaces** prototype.

## Research Questions (authoritative — from the Idea Book entry)

- **RQ1:** How accurately can CNN-based models classify casting product images as defective or non-defective?
- **RQ2:** Does transfer learning improve defective-class recall compared with a custom CNN trained from scratch?
- **RQ3:** Do Grad-CAM and SHAP explanations highlight defect-relevant regions such as holes, burrs, surface deformation, and irregular edges?
- **RQ4:** Which model provides the best balance between inspection accuracy, false-negative reduction, inference speed, and deployment size?
- **RQ5:** Can the selected model be deployed as an explainable web-based casting inspection prototype?

## Suitable Public Kaggle Dataset and Split

**Dataset:** *Casting Product Image Data for Quality Inspection* (Kaggle, by Ravirajsinh Dabhi — "real-life industrial dataset of casting product", submersible-pump impeller castings). It contains roughly **7,300 grayscale images** (300 × 300 augmented set), in **2 classes**: `def_front` (defective) and `ok_front` (OK). It ships with **official `train/` and `test/` folders** (~6,633 train / ~715 test). Exact counts are verified when the data is loaded. The dataset is mildly imbalanced (more defective than OK samples).

**Recommended split:** Keep the official **test** set untouched. Carve **15–20%** of the training set as a **stratified validation** set. For robustness, the experiment may be repeated with three random stratified splits and reported as mean ± standard deviation.

| Split      | Source                                   |
|------------|------------------------------------------|
| Training   | ~80–85% of official train folder         |
| Validation | ~15–20% of official train folder (stratified) |
| Testing    | Official test folder (untouched)         |

## Overall Workflow

1. Define research questions; select the casting defect topic.
2. Collect and inspect the casting dataset.
3. Clean and preprocess (resize 224×224, grayscale→3-channel, normalise).
4. Split: official test untouched, stratified validation from train; prevent leakage.
5. Augment training data (rotation, brightness, contrast, zoom, horizontal flip).
6. Develop the custom CNN baseline.
7. Train EfficientNetV2 and ConvNeXtTiny via transfer learning (freeze → fine-tune).
8. Train, validate, checkpoint with early stopping and LR scheduling.
9. Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, false-negative rate, efficiency.
10. Explain predictions with Grad-CAM and SHAP.
11. Compare models; select the best accuracy/false-negative/efficiency trade-off.
12. Deploy as an explainable web prototype; document reproducibly.
