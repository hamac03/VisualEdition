import streamlit as st
from PIL import Image
import numpy as np
import torch
import tempfile
import os
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image

from grounded_sam2_1_image import segment_objects
from inpainting import inpaint_image


st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Chọn một hình ảnh...", type=["png", "jpg", "jpeg"])


st.sidebar.title("Prompt Input")
prompt_1 = st.sidebar.text_input("Prompt gốc:", "car")
prompt_2 = st.sidebar.text_input("Prompt mới:", "green truck")


num_inference_steps = st.sidebar.slider("Số bước suy luận:", min_value=10, max_value=100, value=50)
button = st.sidebar.button("Edit!")


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Hiển thị ảnh gốc
    # st.subheader("")
    # st.image(image.resize((333, 333)), caption="Origin")

    col1, col2, col3 = st.columns([1, 2, 1])  # Cột giữa lớn hơn
    with col2:  
        st.image(image.resize((512, 512)), caption="Origin Image", use_container_width=True)
    if button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_path = temp_file.name
            image.save(temp_path)  # Lưu ảnh vào file tạm

        try:
            # Thực hiện phân đoạn đối tượng
            prompt_1 = f"{prompt_1}." if not prompt_1.endswith(".") else prompt_1
            mask, image_with_mask = segment_objects(img_path=temp_path, text_prompt=prompt_1)
            print(image_with_mask)
            # Chuyển mask thành ảnh PIL
            mask = (mask[0] * 255).astype(np.uint8)
            pil_mask = Image.fromarray(mask)

            # Thực hiện inpainting
            st.write("**Editting...**")
            res_image = inpaint_image(temp_path, pil_mask, num_inference_steps, prompt_2)

            # Hiển thị kết quả với 3 cột cùng kích thước 512x512
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image.resize((512, 512)), caption="Origin Image", use_container_width =True)
            with col2:
                st.image(Image.fromarray(image_with_mask.astype('uint8')).resize((512, 512)), caption="Mask image", use_container_width =True)

            with col3:
                st.image(res_image.resize((512, 512)), caption="Edited Image", use_container_width =True)
        except:
            st.write("Can't not determine the object. Please check the image or the prompt.")
        finally:
            # Xóa file tạm sau khi sử dụng
            os.remove(temp_path)
