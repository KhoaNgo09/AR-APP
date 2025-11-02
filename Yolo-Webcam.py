# Imports
import streamlit as st
import av
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# -----------------------------
# Streamlit UI
st.title("YOLOv8 + Webcam Realtime với nhãn tiếng Việt")

# -----------------------------
# Hàm vẽ text tiếng Việt
# CẦN ĐẢM BẢO TỆP FONT (VD: arial.ttf) ĐƯỢC ĐẶT TRONG THƯ MỤC DỰ ÁN
def draw_vietnamese_text(img, text, position, font_size=24, color=(255,0,255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 🌟 FIX FONT: Sử dụng đường dẫn tương đối và thêm ngoại lệ kiểm tra
    font_path = "arial.ttf" # <-- Thay đổi tên tệp font nếu cần
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Lỗi: Không tìm thấy tệp font tại {font_path}. Sử dụng font mặc định.")
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# Load model
# 💡 FIX UNPICKLING: Tải bằng tên ("yolov8m") để buộc tải/cache bản sạch.
# CŨNG GIÚP ĐẨY NHANH QUÁ TRÌNH KHỞI TẠO.
model = YOLO("yolov8m")

# -----------------------------
# Class names COCO với nhãn tiếng Việt (Giữ nguyên)
classNames = [
    "Person - Con người", "Bicycle - Xe đạp", "Car - Ô tô", "Motorbike - Xe máy", "Aeroplane - Máy bay",
    "Bus - Xe buýt", "Train - Tàu hỏa", "Truck - Xe tải", "Boat - Thuyền",
    "Traffic Light - Đèn giao thông", "Fire Hydrant - Trụ nước cứu hỏa", "Stop Sign - Biển dừng",
    "Parking Meter - Đồng hồ đỗ xe", "Bench - Ghế dài", "Bird - Chim", "Cat - Mèo",
    "Dog - Chó", "Horse - Ngựa", "Sheep - Cừu", "Cow - Bò", "Elephant - Voi", "Bear - Gấu",
    "Zebra - Ngựa vằn", "Giraffe - Hươu cao cổ", "Backpack - Ba lô", "Umbrella - Ô/Dù",
    "Handbag - Túi xách", "Tie - Cà vạt", "Suitcase - Vali", "Frisbee - Đĩa ném",
    "Skis - Ván trượt tuyết", "Snowboard - Ván trượt tuyết (Một tấm)", "Sports Ball - Bóng thể thao",
    "Kite - Diều", "Baseball Bat - Gậy bóng chày", "Baseball Glove - Găng bóng chày",
    "Skateboard - Ván trượt", "Surfboard - Ván lướt sóng", "Tennis Racket - Vợt Tennis",
    "Bottle - Chai", "Wine Glass - Ly rượu", "Cup - Cốc", "Fork - Nĩa", "Knife - Dao",
    "Spoon - Thìa", "Bowl - Bát", "Banana - Chuối", "Apple - Táo", "Sandwich - Bánh Sandwich",
    "Orange - Cam", "Broccoli - Bông cải xanh", "Carrot - Cà rốt", "Hot Dog - Xúc xích kẹp bánh mì",
    "Pizza - Bánh Pizza", "Donut - Bánh Donut", "Cake - Bánh kem", "Chair - Ghế",
    "Sofa - Ghế Sô Pha", "Potted Plant - Cây cảnh", "Bed - Giường", "Dining Table - Bàn ăn",
    "Toilet - Bồn cầu", "TV Monitor - Tivi/Màn hình", "Laptop - Máy tính xách tay",
    "Mouse - Chuột máy tính", "Remote - Điều khiển", "Keyboard - Bàn phím", "Cell Phone - Điện thoại di động",
    "Microwave - Lò vi sóng", "Oven - Lò nướng", "Toaster - Máy nướng bánh mì", "Sink - Bồn rửa",
    "Refrigerator - Tủ lạnh", "Book - Sách", "Clock - Đồng hồ", "Vase - Bình hoa",
    "Scissors - Kéo", "Teddy Bear - Gấu bông", "Hair Drier - Máy sấy tóc", "Toothbrush - Bàn chải đánh răng"
]

# -----------------------------
# Video transformer YOLO realtime
class YOLOVideoTransformer(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Sử dụng model(img) để dự đoán
        results = model(img) 
        
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf < 0.5:
                    continue
                    
                cls = int(box.cls[0])
                label = f"{classNames[cls]} {conf:.2f}"
                
                img = draw_vietnamese_text(img, label, (x1, y1-25))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------
# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="yolo",
    video_processor_factory=YOLOVideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    # 💡 FIX THREADING/TIMING: Loại bỏ async_processing=True
    # async_processing=True 
)
