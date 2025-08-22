from conf import iou
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO('runs/detect/person-defective/weights/best.pt') #sử dụng model đã được huấn luyện

tracker = DeepSort() #khởi tạo thuật toán deepsort

video_path = 'finaltest.mp4' #đường dẫn tới video
cap = cv2.VideoCapture(video_path) #mở video

count_in = 0 #biến đếm số người vào
count_out = 0 # biến đếm số người ra
track_memory = {} 

#tạo đường line ngang video
ret, frame = cap.read()
frame_height = frame.shape[0]
line_y = frame_height // 2


#tập hợp các id in out tránh bị lặp
counted_in_ids = set()
counted_out_ids = set()


while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:
        break 
    
    results = model(frame, conf=0.5)[0] 
    
    yolo_boxes = []
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
        yolo_boxes.append(((x1, y1, x2, y2), conf))


    tracks = tracker.update_tracks(detections, frame=frame)


    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l,  t, r, b = track.to_ltrb()
#        cx = int((l+r)/2)
        cy = int((t+b)/2)

        track_box = (int(l), int(t), int(r), int(b))
        matched_conf = 0
        max_iou = 0
        for yolo_box, conf in yolo_boxes:
            i = iou(track_box, yolo_box)
            if i > max_iou and i > 0.3:  # chỉ lấy nếu IoU > 0.3
                max_iou = i
                matched_conf = conf
        
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 0, 255), 1)
        cv2.putText(frame, f'ID {track_id} | {matched_conf:.2f}', (int(l), int(t)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if track_id not in track_memory:
            track_memory[track_id] = cy
        else:
            prev_cy = track_memory[track_id]
            if prev_cy < line_y and cy >= line_y and track_id not in counted_in_ids:
                count_in +=1
                counted_in_ids.add(track_id)
            elif prev_cy > line_y and cy <= line_y and track_id not in counted_out_ids:
                count_out +=1
                counted_out_ids.add(track_id)

            track_memory[track_id] = cy

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)
    cv2.putText(frame, f'IN: {count_in}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'OUT: {count_out}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    cv2.imshow('Counting people', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()