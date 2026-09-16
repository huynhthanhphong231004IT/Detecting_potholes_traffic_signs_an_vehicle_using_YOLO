def calculate_distance(clean_class: str, box_h: int, real_heights: dict, focal_length: float) -> tuple[float, tuple[int, int, int]]:
    real_h = real_heights.get("traffic_sign", 0.8)
    for key in real_heights:
        if key in clean_class:
            real_h = real_heights[key]
            break
    pixel_h = max(1, box_h)
    distance = (real_h * focal_length) / pixel_h
    dist_color = (0, 0, 255) if distance < 10.0 else (0, 255, 0)
    return distance, dist_color