#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= TITLE =================
st.title("🗑 Waste Classification System (T1 Model)")

# ================= LOAD T1 MODEL FROM GITHUB =================
MODEL_URL = "https://github.com/Savaira-31/waste-classification-app/raw/main/tl_model.h5"

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    open("tl_model.h5", "wb").write(response.content)
    model = tf.keras.models.load_model("tl_model.h5")
    return model

model = load_model()

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

# ================= CATEGORY MAP =================
def map_category(cls):
    recycle = [
        "aerosol_cans","aluminum_food_cans","aluminum_soda_cans",
        "glass_beverage_bottles","glass_cosmetic_containers","glass_food_jars",
        "plastic_food_containers","plastic_water_bottles",
        "steel_food_cans","cardboard_boxes","cardboard_packaging"
    ]

    reuse = [
        "clothing","shoes","newspaper","magazines","office_paper","paper_cups"
    ]

    if cls in recycle:
        return "♻ Recycle"
    elif cls in reuse:
        return "🔁 Reuse"
    else:
        return "🚯 Reduce / Waste"

# ================= PREDICTION (FIXED FOR T1) =================
def predict(model, img):
    img = img.resize((224, 224))
    img_array = np.array(img)

    img_array = preprocess_input(img_array)  # 🔥 IMPORTANT FOR T1 MODEL
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return class_labels[class_index]

# ================= UI =================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        result = predict(model, img)
        category = map_category(result)

        st.success(f"Predicted Class: {result}")
        st.info(f"Category: {category}")


# In[ ]:




