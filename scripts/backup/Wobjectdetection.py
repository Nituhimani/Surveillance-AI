import cv2
from ultralytics import YOLO

# Assuming the model is already loaded as 'model'
from ultralytics import YOLO
model_path = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\models\new_project\weights\best.pt"
model = YOLO(model_path)

def process_video_frame(frame, model):
    """
    Processes a single video frame using the YOLO model and returns the frame with bounding boxes.
    """
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            # You can set a confidence threshold to filter detections
            if confidence > 0.5: # Example threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def process_video_file(video_path, output_path, model):
    """
    Processes a video file using the YOLO model and saves the output.
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can use other codecs like 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_video_frame(frame, model)
        out.write(processed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_path

# Example usage (assuming 'model' is already loaded)
input_video_path = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\test_videos\bodycam.mp4"
output_video_path = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\results\videos\output_obj2.mp4"
processed_video_path = process_video_file(input_video_path, output_video_path, model)
print(f"Processed video saved to: {processed_video_path}")
