import cv2
import easyocr
import threading
import numpy as np

class AsyncOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.is_busy = False
        self.VALID_SPEEDS = {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "120"}
        self.VALID_TONS = {"1", "1.5", "2", "2.5", "3", "3.5", "5", "7", "8", "10", "12", "15", "18", "20", "25", "30"}
        self.VALID_METERS = {"1.5", "2", "2.2", "2.5", "3", "3.2", "3.5", "3.8", "4", "4.2", "4.5", "5"}

    def process_async(self, cropped_img, box_key, cache_dict):
        if self.is_busy:
            return
        threading.Thread(
            target=self._run_ocr, 
            args=(cropped_img, box_key, cache_dict), 
            daemon=True
        ).start()

    def _run_ocr(self, cropped_img, box_key, cache_dict):
        self.is_busy = True
        try:
            h, w = cropped_img.shape[:2]
            scale = max(2.0, 80.0 / min(h, w))
            resized = cv2.resize(cropped_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([15, 255, 255])) | \
                   cv2.inRange(hsv, np.array([160, 50, 40]), np.array([180, 255, 255]))
            resized[mask > 0] = [255, 255, 255]
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
            results = self.reader.readtext(thresh, allowlist='0123456789.,tmTM', detail=0)
            raw_text = "".join(results).lower().strip().replace(',', '.')
            digits_dots = ''.join([c for c in raw_text if c.isdigit() or c == '.'])
            letters = ''.join([c for c in raw_text if c.isalpha()])
            if digits_dots in ["15", "1.5", "154", "151", "1.54"] and ("t" in letters or "1" in raw_text or "4" in raw_text):
                digits_dots = "1.5"
                letters = "t"
            if digits_dots:
                clean_val = digits_dots.strip('.')
                if 't' in letters or clean_val in self.VALID_TONS:
                    if clean_val in self.VALID_TONS:
                        cache_dict[box_key] = f"{clean_val} t"
                elif 'm' in letters or clean_val in self.VALID_METERS:
                    if clean_val in self.VALID_METERS:
                        cache_dict[box_key] = f"{clean_val} m"
                else:
                    if clean_val in self.VALID_SPEEDS:
                        cache_dict[box_key] = f"{clean_val} km/h"
        except Exception:
            pass
        finally:
            self.is_busy = False