import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Define the list of models and their respective file paths
models = {
    "VGG16": "vgg16_model.h5",
    "VGG19": "vgg19_model.h5",
    "ResNet50": "resnet50_model.h5",
    "DenseNet": "densenet_model.keras"
}

def load_selected_model(model_name):
    return load_model(models[model_name])

def get_last_conv_layer(model_name, model):
    last_conv_layer = None
    if model_name in ["VGG16", "VGG19"]:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer = layer.name
                break
    elif model_name == "ResNet50":
        last_conv_layer = "conv5_block3_out"
    elif model_name == "DenseNet":
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer = layer.name
                break
    return last_conv_layer

def generate_gradcam(model, img_array, last_conv_layer):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), 1e-10)
    return heatmap.numpy()

def create_gradcam_overlay(img_array, heatmap):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[2], img_array.shape[1]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    original_img = img_array[0] * 255
    superimposed_img = jet_heatmap * 0.4 + original_img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

def predict_with_model(model_name, img_array, class_names):
    model = load_selected_model(model_name)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    last_conv_layer = get_last_conv_layer(model_name, model)
    heatmap = generate_gradcam(model, img_array, last_conv_layer)
    superimposed_img = create_gradcam_overlay(img_array, heatmap)
    return predicted_class, confidence, heatmap, superimposed_img

st.title("Grad-CAM Visualization for Deep Learning Models")
model_name = st.selectbox("Select a Model", list(models.keys()))
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
class_names = ['Normal', 'Tuberculosis']

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).resize((224, 224))
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(image_data) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    
    if st.button("Predict and Generate Grad-CAM"):
        predicted_class, confidence, heatmap, superimposed_img = predict_with_model(model_name, img_array, class_names)
        st.write(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(heatmap, cmap='jet')
        ax[0].set_title("Grad-CAM Heatmap")
        ax[0].axis("off")
        ax[1].imshow(superimposed_img)
        ax[1].set_title("Grad-CAM Overlay")
        ax[1].axis("off")
        st.pyplot(fig)
