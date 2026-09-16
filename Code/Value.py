import cv2
import pygame

class Value:
    MODEL_PATH = 'Model/autonomous_vehicle.pt'
    INPUT_VIDEO_PATH = 'Video/Video_03.mp4'
    OUTPUT_VIDEO_PATH = 'Video/Predict_video/output_result_03.mp4'

    TEST_IMAGE_PATH = 'Images/Images_04.png'
    OUTPUT_IMAGE_PATH = 'Images/Predict_img/output_test_04.jpg'
    CONF_THRESHOLD = 0.25
    SPEED_CACHE = {}
    REAL_HEIGHTS = {
        'person': 1.65,
        'car': 1.55,
        'truck': 2.6,
        'bus': 3.2,
        'motorbike': 1.25,
        'bicycle': 1.0,
        'pothole': 0.2,
        'gioi han toc do': 0.8,    
        'cam dung-do xe': 0.8,    
        'cam di nguoc chieu': 0.8, 
        'cam re-quay dau': 0.8,    
        'bien canh bao': 0.7,   
        'bien hieu lenh': 0.8,    
        'bien cam khac': 0.8,      
        'bien phan lan duong': 1.2  
    }

    CAP = cv2.VideoCapture(INPUT_VIDEO_PATH)
    ORIG_WIDTH = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
    ORIG_HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(CAP.get(cv2.CAP_PROP_FPS))
    FOCAL_LENGTH = ORIG_HEIGHT * 1.5
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
    
    INFO = None
    SCREEN_W = None
    SCREEN_H = None
    SCALE = None
    DISP_W = None
    DISP_H = None
    SCREEN = None
    RUNNING = True 

    @classmethod
    def init_pygame_display(cls):
        cls.INFO = pygame.display.Info()
        cls.SCREEN_W, cls.SCREEN_H = cls.INFO.current_w - 100, cls.INFO.current_h - 100
        
        width = cls.ORIG_WIDTH if cls.ORIG_WIDTH > 0 else 1280
        height = cls.ORIG_HEIGHT if cls.ORIG_HEIGHT > 0 else 720
        
        cls.SCALE = min(cls.SCREEN_W / width, cls.SCREEN_H / height)
        cls.DISP_W, cls.DISP_H = int(width * cls.SCALE), int(height * cls.SCALE)
        cls.SCREEN = pygame.display.set_mode((cls.DISP_W, cls.DISP_H))