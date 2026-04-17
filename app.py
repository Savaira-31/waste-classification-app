#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from PIL import Image

# ================= UI =================
st.title("🗑 Waste Classification System")
st.write("🌐 Streamlit Cloud Version (Demo Mode)")

# ================= CATEGORY FUNCTION =================
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

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ================= DEMO PREDICTION =================
    # (Cloud-safe dummy prediction)
    fake_class = "plastic_water_bottles"

    category = map_category(fake_class)

    st.success(f"Predicted Class: {fake_class}")
    st.info(f"Category: {category}")

    st.warning("⚠ Cloud version is demo only (ML model removed for compatibility)")


# In[ ]:




