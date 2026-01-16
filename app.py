import streamlit as st
from PIL import Image
from src.inference import predict

st.title("Touhou(东方) Character Classifier")

uploaded = st.file_uploader("Upload an imag(上传你的东方人物)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    img.save("temp.png")

    from src.gradcam import generate_cam_overlay
    cam, orig, label = generate_cam_overlay("temp.png")

    st.write(f"Prediction: **{label}**")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(orig)
    ax.imshow(cam, cmap='jet', alpha=0.4)
    ax.axis("off")
    st.pyplot(fig)



