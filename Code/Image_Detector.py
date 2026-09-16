import cv2
from Code.Distance_estimator import calculate_distance
from Code.Detect_traffic_light import detect_traffic_light_color  

def process_image(model, ocr_engine, image_path, focal_length, real_heights, speed_cache, output_path, conf_threshold=0.25):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc được ảnh tại {image_path}")
        return

    orig_h, orig_w = frame.shape[:2]
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    current_frame_results = []
    traffic_light_status = None  

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        clean_class = class_name.lower().replace("_", " ").strip()
        
        box_color = (0, 255, 0)
        box_w, box_h = x2 - x1, y2 - y1
        distance, dist_color = calculate_distance(clean_class, box_h, real_heights, focal_length)
        
        display_label_text = class_name
        
        if any(k in clean_class for k in ["gioi han toc do", "speed", "limit", "toc do", "tai trong"]):
            box_color = (0, 165, 255)  
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            box_key = f"{cx // 50}_{cy // 50}"

            if box_key not in speed_cache and box_w > 30 and box_h > 30:
                crop_x1, crop_y1 = max(0, x1 - 2), max(0, y1 - 2)
                crop_x2, crop_y2 = min(orig_w, x2 + 2), min(orig_h, y2 + 2)
                cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                ocr_engine.is_busy = False
                ocr_engine._run_ocr(cropped_img, box_key, speed_cache)

            if box_key in speed_cache:
                current_frame_results.append(speed_cache[box_key])
                display_label_text = f"{class_name} ({speed_cache[box_key]})"

        elif "traffic light" in clean_class or "den giao thong" in clean_class:
            box_color = (255, 0, 0)
            detected_status = detect_traffic_light_color(frame, x1, y1, x2, y2)
            if detected_status:
                traffic_light_status = detected_status

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        label = f"{display_label_text}: {conf * 100:.1f}%"
        cv2.putText(frame, label, (x1, max(y1 - 22, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(y1 - 22, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        dist_label = f"Dist: {distance:.1f}m"
        cv2.putText(frame, dist_label, (x1, max(y1 - 5, 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, dist_label, (x1, max(y1 - 5, 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dist_color, 1, cv2.LINE_AA)
    unique_results = list(set(current_frame_results))
    if unique_results:
        overlay_y = 35
        for item in unique_results:
            if "km/h" in item:
                display_text = f"Toc do: {item}"
            elif "t" in item:
                display_text = f"Tai trong: {item}"
            elif "m" in item:
                display_text = f"Chieu cao: {item}"
            else:
                display_text = f"Bien bao: {item}"

            cv2.putText(frame, display_text, (20, overlay_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, display_text, (20, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            overlay_y += 35
    if traffic_light_status:
        light_text = f"Trang thai: {traffic_light_status}"
        (lt_w, lt_h), lt_base = cv2.getTextSize(light_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    
        if "do" in traffic_light_status:
            text_color = (0, 0, 255)    
        elif "vang" in traffic_light_status:
            text_color = (0, 255, 255) 
        else:
            text_color = (0, 255, 0)    

        box_x = orig_w - lt_w - 35
        cv2.rectangle(frame, (box_x, 20), (orig_w - 15, 20 + lt_h + lt_base + 20), (0, 0, 0), -1)
        cv2.putText(frame, light_text, (box_x + 10, 20 + lt_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    cv2.imwrite(output_path, frame)
    
    cv2.imshow("Image Test - Traffic Sign & Distance Detector", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()