
import cv2
import gradio as gr
from ultralytics import YOLO
import os

# Tải mô hình
model = YOLO('yolov8n.pt')

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = "output_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track và nhận diện
        results = model.track(frame, persist=True, classes=vehicle_classes, verbose=False)
        res_plotted = results[0].plot()
        out.write(res_plotted)

    cap.release()
    out.release()
    return output_path

demo = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(),
    outputs=gr.Video(),
    title="YOLOv8 Vehicle Counter"
)

if __name__ == "__main__":
    demo.launch()
