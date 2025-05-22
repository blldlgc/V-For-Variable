from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import tempfile
import os
import base64
from ultralytics import YOLO
from tensorflow.keras.models import load_model

app = FastAPI()

# 🔍 Modellerin yüklenmesi
segment_model = YOLO("yolov8_model.pt")  # Segmentasyon modeli
classifier_model = load_model("eye_pad_model.keras")  # Keras sınıflandırma modeli
class_names = ["LEKELI", "SAGLAM"]  # Eğitildiği sıraya göre güncelle

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

def ellipse_fit_error(contour, ellipse):
    ellipse_points = cv2.ellipse2Poly(
        center=(int(ellipse[0][0]), int(ellipse[0][1])),
        axes=(int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
        angle=int(ellipse[2]),
        arcStart=0,
        arcEnd=360,
        delta=5
    )
    min_len = min(len(contour), len(ellipse_points))
    distances = [np.linalg.norm(contour[i][0] - ellipse_points[i]) for i in range(min_len)]
    return float(np.mean(distances))

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    print(f"📥 Yeni istek alındı: {file.filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Geçersiz görsel. cv2.imread() None döndü.")
        return JSONResponse(content={"error": "Geçersiz görsel: Dosya okunamadı"}, status_code=400)
    
    img = cv2.resize(img, (640, 640))
    results = segment_model(img, conf=0.3, task="segment")[0]
    masks = results.masks
    if masks is None:
        print("⚠️ Segmentasyon maskesi bulunamadı.")
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
        "measurements": {},
        "std_dev": None,
        "fit_error_ic": None,
        "fit_error_dis": None,
        "keras_prediction": None,
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

        # Fit error hesapla
        for label in ["goz_pedi_ic", "goz_pedi_dis"]:
            try:
                mask_idx = list(results.names.keys())[list(results.names.values()).index(label)]
                mask = (masks.data[mask_idx].cpu().numpy() * 255).astype(np.uint8)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    if len(cnt) >= 5:
                        err = ellipse_fit_error(cnt, mask_info[label])
                        if label == "goz_pedi_ic":
                            json_out["fit_error_ic"] = err
                        else:
                            json_out["fit_error_dis"] = err
            except:
                continue

        # 🔍 Keras sınıflandırma
        try:
            keras_img = cv2.resize(img, (224, 224))
            keras_img = keras_img.astype(np.float32) / 255.0
            keras_input = np.expand_dims(keras_img, axis=0)

            prediction = classifier_model.predict(keras_input)
            predicted_class = class_names[np.argmax(prediction)]
            confidence_score = float(np.max(prediction))

            json_out["keras_prediction"] = {
                "class": predicted_class,
                "confidence": confidence_score
            }

            cv2.putText(drawn_img, f"Keras: {predicted_class} ({confidence_score:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        except Exception as e:
            print(f"⚠️ Keras tahmini hatası: {e}")
            json_out["keras_prediction"] = {
                "error": "Keras modeli çalıştırılamadı"
            }

        drawn_bgr = cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', drawn_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        json_out["visual_base64"] = img_base64
    else:
        print("⚠️ Gerekli etiketler eksik: 'goz_pedi_ic' veya 'goz_pedi_dis'")

    os.remove(image_path)
    print("✅ İşlem tamamlandı.")
    return JSONResponse(content=json_out)
