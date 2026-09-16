import cv2
import numpy as np

def detect_traffic_light_color(frame, x1, y1, x2, y2):
    orig_h, orig_w = frame.shape[:2]
    crop_x1, crop_y1 = max(0, x1), max(0, y1)
    crop_x2, crop_y2 = min(orig_w, x2), min(orig_h, y2)
    
    light_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if light_crop.size == 0 or light_crop.shape[0] < 5 or light_crop.shape[1] < 5:
        return None

    # 1. Chuyển sang Float để tránh tràn số khi tính toán
    img_float = light_crop.astype(np.float32) + 1.0 # Tránh chia cho 0
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    
    # 2. Khôi phục màu bị lóa bằng Chromaticity (Tỷ lệ màu chuẩn hóa)
    # Tỷ lệ này KHÔNG DÙNG HSV nên không sợ Saturation = 0 ở tâm lóa
    total_intensity = r + g + b
    r_norm = r / total_intensity
    g_norm = g / total_intensity
    b_norm = b / total_intensity

    # 3. Tạo Mask lọc vùng phát sáng (Loại bỏ nền trời, hộp đèn xám/đen)
    hsv = cv2.cvtColor(light_crop, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2]
    
    # Chỉ xét những pixel thực sự phát sáng (V > 120)
    bright_mask = v_chan > 120

    # 4. Phân loại màu sắc dựa trên Chromaticity & Tỷ lệ chênh lệch kênh
    # Đèn XANH LED: g_norm chiếm ưu thế HOẶC (G > R và G > B)
    mask_green = bright_mask & (
        ((g_norm > 0.38) & (g_norm > r_norm * 1.05)) | 
        ((g > r * 1.1) & (g > b * 0.9))
    )

    # Đèn ĐỎ: r_norm vượt trội so với g_norm
    mask_red = bright_mask & (
        ((r_norm > 0.42) & (r_norm > g_norm * 1.2)) | 
        ((r > g * 1.3) & (r > b * 1.2))
    )

    # Đèn VÀNG: R và G cân bằng nhau nhưng đều lớn hơn B rõ rệt
    mask_yellow = bright_mask & (
        (r_norm > 0.35) & (g_norm > 0.35) & (abs(r_norm - g_norm) < 0.12) & (r_norm > b_norm * 1.3)
    )

    # 5. Đếm số lượng pixel thỏa mãn từng màu
    count_red = np.sum(mask_red)
    count_yellow = np.sum(mask_yellow)
    count_green = np.sum(mask_green)

    scores = {
        "Den do": count_red,
        "Den vang": count_yellow,
        "Den xanh la": count_green
    }

    dominant_color = max(scores, key=scores.get)
    max_score = scores[dominant_color]

    # Ngưỡng lọc nhiễu: Cần ít nhất 0.8% số pixel trong bounding box sáng đúng màu
    if max_score < (light_crop.shape[0] * light_crop.shape[1] * 0.008):
        return None

    return dominant_color