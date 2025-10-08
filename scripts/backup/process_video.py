from ultralytics import YOLO
import cv2
import json

def process_video(input_video_path, output_video_path, output_json_path, model_path="/content/best_activity.pt", conf_threshold=0.5):
    """
    Processes a video using a YOLO model for object detection and saves the output.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str): The path to save the output video file.
        output_json_path (str): The path to save the output JSON file.
        model_path (str): The path to the trained YOLO model.
        conf_threshold (float): The confidence threshold for detections.
    """
    # Load the trained model
    model = YOLO(model_path)

    # OpenCV video reader
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer with a web-compatible codec (H.264)
    # Using 'avc1' which corresponds to H.264 for MP4 files. 'mp4v' is often not supported by browsers.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    frame_id = 0
    results_list = []  # for JSON

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, conf=conf_threshold)

        # Process results and draw bounding boxes
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # Save detection for JSON
                results_list.append({
                    "frame": frame_id,
                    "label": label,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

                # Draw box on frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write processed frame
        out.write(frame)
        frame_id += 1

    # Release resources
    cap.release()
    out.release()

    # Save detections to JSON
    with open(output_json_path, "w") as f:
        json.dump(results_list, f, indent=4)

    print(f"Processing complete! Output video: {output_video_path}, JSON: {output_json_path}")

if __name__ == "__main__":
    # Example usage:
    # Ensure you have a video and a model file at these paths or update them.
    input_video = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\test_videos\bodycam.mp4"
    output_video = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\results\videos\output_act2.mp4"
    output_json = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\results\vid_json\output_act2.json"
    model_path = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\models\suspicious_model4\weights\best.pt"

    process_video(input_video, output_video, output_json, model_path)