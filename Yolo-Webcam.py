from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# -----------------------------
# Hàm vẽ text tiếng Việt
def draw_vietnamese_text(img, text, position, font_size=24, color=(255,0,255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# -----------------------------
# Load YOLO model
model = YOLO("yolov8m.pt")

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
# Video transformer xử lý YOLO realtime
class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                cx,cy = (x1+x2)//2, (y1+y2)//2
                frame_h, frame_w,_ = img.shape
                center_zone_x = (int(frame_w*0.3), int(frame_w*0.7))
                center_zone_y = (int(frame_h*0.3), int(frame_h*0.7))
                if not (center_zone_x[0]<cx<center_zone_x[1] and center_zone_y[0]<cy<center_zone_y[1]):
                    continue
                cls = int(box.cls[0])
                label = f"{classNames[cls]} {conf:.2f}"
                img = draw_vietnamese_text(img, label, (x1, y1-25))
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)

        return img

# -----------------------------
st.title("YOLOv8 + Webcam Realtime trên Streamlit")

webrtc_streamer(
    key="example",
    video_transformer_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

