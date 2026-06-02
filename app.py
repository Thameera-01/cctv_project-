import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture('17604987-hd_1920_1080_30fps.mp4')

count_line_y = 400 
vehicle_counter = 0
counted_ids = set() 

while cap.isOpened():
    success, frame = cap.read()

    if success:
        
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):

                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if count_line_y - 10 < cy < count_line_y + 10:
                    if id not in counted_ids:
                        vehicle_counter += 1
                        counted_ids.add(id)
                        cv2.circle(annotated_frame, (cx, cy), 10, (0, 255, 0), -1)

        cv2.line(annotated_frame, (0, count_line_y), (frame.shape[1], count_line_y), (0, 0, 255), 3)
        
        cv2.putText(annotated_frame, f'Vehicles: {vehicle_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("YOLOv8 Vehicle Counting", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()