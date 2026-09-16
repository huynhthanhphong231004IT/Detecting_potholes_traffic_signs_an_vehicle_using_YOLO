import cv2
import pygame
import gc
import torch
from Code.Distance_estimator import calculate_distance
from Code.Detect_traffic_light import detect_traffic_light_color


def process_video(model, ocr_engine, cap, out, screen, clock, fps, disp_size, orig_size, focal_length, real_heights, conf_threshold=0.25):
    disp_w, disp_h = disp_size
    orig_w, orig_h = orig_size
    speed_cache = {}
    running = True
    current_stable_status = None  
    pending_status = None         
    pending_count = 0             
    missing_count = 0           
    
    CONFIRM_FRAMES = 4         
    MAX_MISSING_FRAMES = 10       

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with torch.no_grad():
            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]

        current_frame_results = []
        traffic_light_boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            clean_class = class_name.lower().replace("_", " ").strip()
            
            box_color = (0, 255, 0)
            box_w, box_h = x2 - x1, y2 - y1

            distance, dist_color = calculate_distance(clean_class, box_h, real_heights, focal_length)
            
            if any(k in clean_class for k in ["gioi han toc do", "speed", "limit", "toc do", "tai trong"]):
                box_color = (0, 215, 255)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                box_key = f"{cx // 50}_{cy // 50}"

                if len(speed_cache) > 100:
                    speed_cache.clear()

                if box_key not in speed_cache and box_w > 30 and box_h > 30:
                    crop_x1, crop_y1 = max(0, x1 - 2), max(0, y1 - 2)
                    crop_x2, crop_y2 = min(orig_w, x2 + 2), min(orig_h, y2 + 2)
                    cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    ocr_engine.process_async(cropped_img, box_key, speed_cache)

                if box_key in speed_cache:
                    current_frame_results.append(speed_cache[box_key])
            
            elif "traffic light" in clean_class or "den giao thong" in clean_class:
                box_color = (255, 0, 0)
                area = box_w * box_h
                traffic_light_boxes.append({
                    'coords': (x1, y1, x2, y2),
                    'area': area
                })

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"{class_name}: {conf * 100:.1f}%"
            cv2.putText(frame, label, (x1, max(y1 - 22, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            dist_label = f"Dist: {distance:.1f}m"
            cv2.putText(frame, dist_label, (x1, max(y1 - 5, 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dist_color, 2)
        raw_status = None
        if traffic_light_boxes:
            largest_light = max(traffic_light_boxes, key=lambda b: b['area'])
            lx1, ly1, lx2, ly2 = largest_light['coords']
            raw_status = detect_traffic_light_color(frame, lx1, ly1, lx2, ly2)
        if raw_status is not None:
            missing_count = 0 
            
            if raw_status == pending_status:
                pending_count += 1
            else:
                pending_status = raw_status
                pending_count = 1

            if pending_count >= CONFIRM_FRAMES:
                current_stable_status = pending_status
        else:
            missing_count += 1
            if missing_count >= MAX_MISSING_FRAMES:
                current_stable_status = None
                pending_status = None
                pending_count = 0
        unique_results = list(set(current_frame_results))
        if unique_results:
            overlay_y = 40
            for item in unique_results:
                if "km/h" in item:
                    display_text = f"Bien bao toc do: {item}"
                elif "t" in item:
                    display_text = f"Bien bao tai trong: {item}"
                elif "m" in item:
                    display_text = f"Bien bao chieu cao: {item}"
                else:
                    display_text = f"Gia tri bien bao: {item}"
                (tw, th), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (10, overlay_y - th - 10), (15 + tw + 10, overlay_y + baseline + 5), (0, 0, 0), -1)
                cv2.putText(frame, display_text, (15, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                overlay_y += th + 20
        if current_stable_status:
            light_text = f"Trang thai: {current_stable_status}"
            (lt_w, lt_h), lt_base = cv2.getTextSize(light_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            if "do" in current_stable_status:
                text_color = (0, 0, 255)  
            elif "vang" in current_stable_status:
                text_color = (0, 255, 255) 
            else:
                text_color = (0, 255, 0)     
            box_x = orig_w - lt_w - 35
            cv2.rectangle(frame, (box_x, 20), (orig_w - 15, 20 + lt_h + lt_base + 20), (0, 0, 0), -1)
            cv2.putText(frame, light_text, (box_x + 10, 20 + lt_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        out.write(frame)
        resized_frame = cv2.resize(frame, (disp_w, disp_h))
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(frame_surface, (0, 0))
        pygame.display.update()
        clock.tick(fps)
        del results
        gc.collect()