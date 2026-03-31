import os
import cv2
import torch
import numpy as np
import torchvision
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from groundingdino.util.inference import load_model, predict
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
import groundingdino.datasets.transforms as T
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DINO_DEVICE = torch.device("cpu")
SAM_DEVICE = torch.device("mps")

def init_engine(gd_cfg, gd_ckpt, sam_cfg, sam_ckpt):
    print("start！！！！！！！！！！！！！！！！！！！！")
    gd_model = load_model(gd_cfg, gd_ckpt, device="cpu")
    gd_model.to(DINO_DEVICE)

    sam_model = build_sam2(sam_cfg, sam_ckpt, device=SAM_DEVICE)
    sam_predictor = SAM2ImagePredictor(sam_model)
    return gd_model, sam_predictor

def load_image_for_dino(image_path):
    image_source = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_source, None)
    return np.array(image_source), image_transformed

def start_auto_labeling(input_folder, output_folder, prompt, box_threshold=0.35):
    gd_cfg = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gd_ckpt = "checkpoints/groundingdino_swint_ogc.pth"
    sam_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_ckpt = "checkpoints/sam2.1_hiera_large.pt"

    gd_model, sam_predictor = init_engine(gd_cfg, gd_ckpt, sam_cfg, sam_ckpt)

    in_path = Path(input_folder)
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in in_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    print(f"开始处理 {len(image_files)} 张图片，当前 Prompt: {prompt}")

    for img_file in tqdm(image_files):
        img_source, img_tensor = load_image_for_dino(img_file)
        h, w, _ = img_source.shape

        boxes, logits, phrases = predict(
            model=gd_model, image=img_tensor, caption=prompt,
            box_threshold=box_threshold, text_threshold=0.25, device=DINO_DEVICE
        )
        
        if len(boxes) == 0: continue

        boxes_xyxy_norm = box_cxcywh_to_xyxy(boxes)
        
        boxes_xyxy_pixel = boxes_xyxy_norm * torch.Tensor([w, h, w, h])

        keep_indices = torchvision.ops.nms(boxes_xyxy_pixel, logits, iou_threshold=0.5)
        
        boxes_final = boxes_xyxy_pixel[keep_indices]
        logits_final = logits[keep_indices]
        phrases_final = [phrases[i] for i in keep_indices]

        sam_predictor.set_image(img_source)
        masks, scores, _ = sam_predictor.predict(
            box=boxes_final.to(SAM_DEVICE),
            multimask_output=False
        )

        res_img = img_source.copy()
        for i, mask in enumerate(masks):
            color = np.random.randint(0, 255, (3,)).tolist()
            mask_bool = (mask[0] > 0).astype(np.uint8)
            for c in range(3):
                res_img[:, :, c] = np.where(mask_bool == 1,
                                            res_img[:, :, c] * 0.5 + color[c] * 0.5,
                                            res_img[:, :, c])
            
            x1, y1, x2, y2 = boxes_final[i].int().tolist()
            cv2.rectangle(res_img, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{phrases_final[i]} {logits_final[i]:.2f}"
            cv2.putText(res_img, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        final_save_path = out_path / f"fixed_{img_file.name}"
        Image.fromarray(res_img).save(final_save_path)

if __name__ == "__main__":
    INPUT_DIR = "assets"
    OUTPUT_DIR = "auto_labels"
    PROMPT = "car . tire . person ."
    THRESHOLD = 0.3

    start_auto_labeling(INPUT_DIR, OUTPUT_DIR, PROMPT, THRESHOLD)
