from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import tempfile
import os
import base64
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8_model.pt")
sapma_orani = 10

def get_ellipse_point(center, axes, angle_deg, direction):
    angle_rad = np.deg2rad(angle_deg)
    cx, cy = center
    a, b = axes[0] / 2, axes[1] / 2
    theta = {'top': -90, 'bottom': 90, 'left': 180, 'right': 0}[direction]
    theta_rad = np.deg2rad(theta)
    x = a * np.cos(theta_rad)
    y = b * np.sin(theta_rad)
    xr = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    yr = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return int(cx + xr), int(cy + yr)

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    img = cv2.imread(image_path)
    if img is None:
        return JSONResponse(content={"error": "Geçersiz görsel"}, status_code=400)

    results = model(image_path, conf=0.3, task="segment")[0]
    masks = results.masks
    if masks is None:
        return JSONResponse(content={"error": "Segmentasyon maskesi bulunamadı"}, status_code=400)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    drawn_img = img_rgb.copy()
    mask_info = {}

    for i, mask in enumerate(masks.data):
        cls = int(results.boxes.cls[i].item())
        label = results.names[cls]
        binary = (mask.cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                mask_info[label] = ellipse

    json_out = {
        "image": file.filename,
        "threshold_px": sapma_orani,
        "measurements": {},
        "std_dev": None,
        "status": "undefined",
        "visual_base64": None
    }

    if "goz_pedi_ic" in mask_info and "goz_pedi_dis" in mask_info:
        e_ic = mask_info["goz_pedi_ic"]
        e_dis = mask_info["goz_pedi_dis"]
        directions = ["top", "bottom", "left", "right"]
        distances = []

        cv2.ellipse(drawn_img, e_ic, (0, 255, 0), 2)
        cv2.ellipse(drawn_img, e_dis, (255, 0, 0), 2)

        for dir in directions:
            p1 = get_ellipse_point(e_ic[0], e_ic[1], e_ic[2], dir)
            p2 = get_ellipse_point(e_dis[0], e_dis[1], e_dis[2], dir)
            dist = float(np.linalg.norm(np.array(p1) - np.array(p2)))
            distances.append(dist)
            cv2.line(drawn_img, p1, p2, (0, 255, 255), 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(drawn_img, f"{dir[0].upper()}: {dist:.1f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            json_out["measurements"][dir] = {
                "distance_px": dist,
                "point_ic": {"x": p1[0], "y": p1[1]},
                "point_dis": {"x": p2[0], "y": p2[1]}
            }

        std_dev = float(np.std(distances))
        json_out["std_dev"] = std_dev
        json_out["status"] = "hatalı" if std_dev > sapma_orani else "hatasız"

        # Görseli base64 olarak encode et
        drawn_bgr = cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', drawn_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        json_out["visual_base64"] = img_base64

    os.remove(image_path)
    return JSONResponse(content=json_out)
