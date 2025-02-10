import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import matplotlib.pyplot as plt


def segment_objects(img_path, text_prompt, grounding_model="IDEA-Research/grounding-dino-tiny", sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt", sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml", force_cpu=False):
    # Thiết lập thiết bị
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"

    # Bật chế độ tự động chọn kiểu dữ liệu để tăng hiệu suất
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Tạo mô hình SAM2
    sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Load Grounding DINO
    processor = AutoProcessor.from_pretrained(grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(device)

    # Đọc ảnh
    image = Image.open(img_path)
    sam2_predictor.set_image(np.array(image.convert("RGB")))

    # Xử lý đầu vào cho Grounding DINO
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
    )

    # Trích xuất bounding boxes và mask từ SAM2
    input_boxes = results[0]["boxes"].cpu().numpy()
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None, point_labels=None, box=input_boxes, multimask_output=False
    )

    # Chuyển định dạng mask
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    class_ids = np.array(list(range(len(class_names))))

    labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences)]

    # Annotate hình ảnh
    img = cv2.imread(img_path)
    detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)

    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)


    # Hiển thị kết quả với Matplotlib
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_frame_rgb)
    plt.axis("off")
    plt.show()

    return masks, annotated_frame_rgb



# Chạy thử
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", required=True, help="Path to input image")
    parser.add_argument("--text-prompt", required=True, help="Object detection prompt")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="./configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--force-cpu", action="store_true")

    args = parser.parse_args()
    segment_objects(
        args.img_path,
        args.text_prompt,
        args.grounding_model,
        args.sam2_checkpoint,
        args.sam2_model_config,
        args.force_cpu
    )
