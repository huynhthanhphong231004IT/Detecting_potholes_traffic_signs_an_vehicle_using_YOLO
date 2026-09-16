import cv2
import pygame

from ultralytics import YOLO
from Code.AsyncOCR import AsyncOCR as OCR
from Code.Value import Value as Val
from Code.Detector import process_video
from Code.Image_Detector import process_image

ocr_engine = OCR()
model = YOLO(Val.MODEL_PATH)

if __name__ == "__main__":
    pygame.init()
    Val.init_pygame_display()
    cap = Val.CAP
    
    # if cap.isOpened():
    #     out = cv2.VideoWriter(Val.OUTPUT_VIDEO_PATH, Val.FOURCC, Val.FPS, (Val.ORIG_WIDTH, Val.ORIG_HEIGHT))
    #     pygame.display.set_caption("Real-Time Ultra Speed Traffic Sign & Distance Detector")
    #     clock = pygame.time.Clock()

    #     process_video(
    #         model=model,
    #         ocr_engine=ocr_engine,
    #         cap=cap,
    #         out=out,
    #         screen=Val.SCREEN,
    #         clock=clock,
    #         fps=Val.FPS,
    #         disp_size=(Val.DISP_W, Val.DISP_H),
    #         orig_size=(Val.ORIG_WIDTH, Val.ORIG_HEIGHT),
    #         focal_length=Val.FOCAL_LENGTH,
    #         real_heights=Val.REAL_HEIGHTS,
    #         # speed_cache=Val.SPEED_CACHE,
    #         conf_threshold=Val.CONF_THRESHOLD
    #     )
    #     cap.release()
    #     out.release()
    # else:
    #     print(f"Cảnh báo: Không mở được video tại {Val.INPUT_VIDEO_PATH}")

    test_img = cv2.imread(Val.TEST_IMAGE_PATH)
    if test_img is not None:
        process_image(
            model=model,
            ocr_engine=ocr_engine,
            image_path=Val.TEST_IMAGE_PATH,
            focal_length=Val.FOCAL_LENGTH,
            real_heights=Val.REAL_HEIGHTS,
            speed_cache=Val.SPEED_CACHE,
            output_path=Val.OUTPUT_IMAGE_PATH,
            conf_threshold=Val.CONF_THRESHOLD
        )
    else:
        print(f"Cảnh báo: Không tìm thấy ảnh tại {Val.TEST_IMAGE_PATH}")

    pygame.quit()