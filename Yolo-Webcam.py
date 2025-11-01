from ultralytics import YOLO
import cv2
import cvzone
import math

import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Lệnh dùng Code hỗ trợ display TViệt
def draw_vietnamese_text(img1, text, position, font_size=24, color=(255, 255, 255)):
    # Chuyển ảnh OpenCV sang Pillow (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img_pil)

    # Dùng font có hỗ trợ tiếng Việt (nhớ để file .ttf trong cùng thư mục)
    font = ImageFont.truetype("arial.ttf", font_size)  # hoặc tahoma.ttf, times.ttf

    draw.text(position, text, font=font, fill=color)

    # Chuyển ảnh về lại OpenCV (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# cap = cv2.VideoCapture("../Videos/ppe-2-1.mp4")

model = YOLO("../Yolo-Weights/yolov8m.pt")

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

while True:
    success, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1,y1,x2,y2  = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2-x1, y2-y1

            cvzone.cornerRect(img,(x1,y1,w,h))
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # Chỉ hiện vật có độ tin cậy cao
            # 🎯 1. Bỏ qua nếu độ tin cậy thấp hơn 0.5
            if conf < 0.5    :
                continue

            # 🎯 2. Tính tâm của vật thể
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 🎯 3. Lấy kích thước khung hình (chỉ cần lấy 1 lần ở vòng đầu)
            frame_h, frame_w, _ = img.shape

            # 🎯 4. Xác định “vùng trung tâm” (ví dụ 40% giữa khung hình)
            center_zone_x = (int(frame_w * 0.3), int(frame_w * 0.7))
            center_zone_y = (int(frame_h * 0.3), int(frame_h * 0.7))

            # 🎯 5. Chỉ nhận vật nếu tâm nằm trong vùng trung tâm
            if not (center_zone_x[0] < cx < center_zone_x[1] and center_zone_y[0] < cy < center_zone_y[1]):
                continue
            # Class Name
            cls = int(box.cls[0])
            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=3)

            label = f"{classNames[cls]} {conf:.2f}"
            img = draw_vietnamese_text(img, label, (x1, y1 - 25), font_size=24, color=(255, 0, 255))


    cv2.imshow("Image", img)
    cv2.waitKey(1)