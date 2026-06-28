# Streamlit web app for explainable casting defect inspection.
# It loads the trained custom CNN, predicts defective vs OK for an uploaded
# casting image, and shows a Grad-CAM heatmap that explains where the model looked.
#
# Run locally:   pip install -r requirements.txt   then   streamlit run streamlit_app.py
# Deploy:        push this folder to GitHub, then point Streamlit Community Cloud at it.
import os
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.keras")

st.set_page_config(page_title="Casting Defect Inspector", page_icon="check", layout="centered")


# @st.cache_resource keeps the model in memory across reruns, so we load it once
# instead of on every interaction (loading a model is slow).
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def last_conv_layer_name(model):
    # Grad-CAM needs the last convolutional layer; we find it by scanning backwards.
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None


def preprocess(pil_img):
    # The model expects 224x224 RGB pixels in the 0-255 range; it rescales internally.
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img).astype("float32")


def grad_cam(model, arr, conv_name):
    # Standard Grad-CAM: weight the last conv feature maps by the gradient of the
    # defective score, then keep the positive part and normalise to 0-1.
    grad_model = keras.models.Model(model.inputs,
                                    [model.get_layer(conv_name).output, model.output])
    x = tf.convert_to_tensor(arr[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], [IMG_SIZE, IMG_SIZE])[..., 0].numpy()


def overlay(arr, cam, alpha=0.5):
    heat = cm.jet(cam)[..., :3]          # turn the heatmap into RGB colours
    base = arr / 255.0
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


st.title("Casting Product Defect Inspector")
st.write(
    "Upload a casting image. The model predicts whether the part is defective or "
    "acceptable, and the heatmap shows which region drove the decision."
)

model = load_model()
conv_name = last_conv_layer_name(model)

file = st.file_uploader("Upload a casting image", type=["jpg", "jpeg", "png", "bmp"])
if file is not None:
    pil_img = Image.open(file)
    arr = preprocess(pil_img)
    prob = float(model.predict(arr[None], verbose=0).ravel()[0])   # P(defective)
    label = "DEFECTIVE" if prob >= 0.5 else "OK"
    confidence = prob if prob >= 0.5 else 1.0 - prob

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)
    with col2:
        if conv_name:
            st.image(overlay(arr, grad_cam(model, arr, conv_name)),
                     caption="Grad-CAM", use_container_width=True)

    if label == "DEFECTIVE":
        st.error(f"Prediction: {label}  (confidence {confidence:.1%})")
    else:
        st.success(f"Prediction: {label}  (confidence {confidence:.1%})")
    st.caption(
        f"P(defective) = {prob:.3f}, decision threshold = 0.50. "
        "This is a decision-support tool and a human should confirm flagged parts."
    )
