#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
from PIL import Image
import numpy as np

st.title("🗑 Waste Classification System (Demo Mode)")

st.write("⚠ Streamlit Cloud Demo Version (No ML model)")

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

    reduce = [
        "food_waste","eggshells","coffee_grounds","tea_bags",
        "plastic_trash_bags","plastic_straws","plastic_shopping_bags"
    ]

    if cls in recycle:
        return "♻ Recycle"
    elif cls in reuse:
        return "🔁 Reuse"
    else:
        return "🚯 Reduce / Waste"

# ================= UI =================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ================= SMART RULE-BASED “FAKE PREDICTION” =================
    # (This is deterministic demo logic)

    img_array = np.array(img)
    brightness = np.mean(img_array)

    # simple heuristic mapping (demo purpose)
    if brightness < 80:
        fake_class = "food_waste"
    elif brightness < 140:
        fake_class = "cardboard_boxes"
    else:
        fake_class = "plastic_water_bottles"

    category = map_category(fake_class)

    st.success(f"Predicted Class: {fake_class}")
    st.info(f"Category: {category}")

    st.warning("⚠ Demo mode: real ML model removed for cloud compatibility")


# In[ ]:




