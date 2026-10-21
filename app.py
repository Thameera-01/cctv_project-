import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture('17604987-hd_1920_1080_30fps.mp4')

count_line_y = 400 
track_history = {} 
counted_ids = set() 

counts = {
    'In': {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0},
    'Out': {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
}
class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])
    
    overlay = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            class_name = class_names.get(cls, 'Unknown')

            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))
            
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            for i in range(1, len(track_history[track_id])):
                cv2.line(frame, track_history[track_id][i-1], track_history[track_id][i], (0, 255, 255), 2)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id} {class_name}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if track_id not in counted_ids and len(track_history[track_id]) >= 2:
                prev_y = track_history[track_id][-2][1] 
                curr_y = track_history[track_id][-1][1]

                if prev_y < count_line_y and curr_y >= count_line_y:
                    counts['Out'][class_name] += 1
                    counted_ids.add(track_id)
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1) 

                elif prev_y > count_line_y and curr_y <= count_line_y:
                    counts['In'][class_name] += 1
                    counted_ids.add(track_id)
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 0), -1) 

    cv2.line(frame, (0, count_line_y), (frame.shape[1], count_line_y), (255, 0, 255), 3)

    cv2.rectangle(overlay, (20, 20), (350, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame) 

    cv2.putText(frame, "TRAFFIC MONITOR (IN / OUT)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (30, 65), (330, 65), (255, 255, 255), 1)

    y_pos = 100
    for v_class in ['Car', 'Motorcycle', 'Bus', 'Truck']:
        in_count = counts['In'][v_class]
        out_count = counts['Out'][v_class]
        cv2.putText(frame, f"{v_class}:", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"IN: {in_count} | OUT: {out_count}", (160, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += 40

    cv2.imshow("Advanced AI Traffic Management System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
