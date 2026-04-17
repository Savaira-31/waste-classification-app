#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import tensorflow as tf
import gdown
from PIL import Image

# ================= LOAD CNN MODEL =================
url = "https://drive.google.com/uc?id=1SzFNQ8N9Rs7Yo0x6sCt4eBR_9THtnWNw"
output = "cnn_model.h5"

gdown.download(url, output, quiet=True)
cnn_model = tf.keras.models.load_model("cnn_model.h5")

# ================= LOAD TL MODEL =================
tl_model = tf.keras.models.load_model("tl_model.h5")

# ================= CLASS LABELS =================
class_labels = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans",
    "cardboard_boxes", "cardboard_packaging", "clothing",
    "coffee_grounds", "disposable_plastic_cutlery", "eggshells",
    "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper",
    "paper_cups", "plastic_cup_lids", "plastic_detergent_bottles",
    "plastic_food_containers", "plastic_shopping_bags",
    "plastic_soda_bottles", "plastic_straws", "plastic_trash_bags",
    "plastic_water_bottles", "shoes", "steel_food_cans",
    "styrofoam_cups", "styrofoam_food_containers", "tea_bags"
]

# ================= PREDICTION =================
def predict(model, img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return class_labels[np.argmax(prediction)]

# ================= STREAMLIT UI =================
st.title("🗑 Waste Classification System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

model_choice = st.radio("Choose Model", ["CNN Model", "Pretrained Model"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    if st.button("Predict"):
        if model_choice == "CNN Model":
            result = predict(cnn_model, img)
        else:
            result = predict(tl_model, img)

        st.success(f"Predicted Class: {result}")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "app.ipynb"')


# In[ ]:




