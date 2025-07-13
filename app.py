
import gradio as gr
from ultralytics import YOLO
import os

d_data = "best.pt"

model = YOLO(d_data)

def detect_fire_image(img_path):
    results = model.predict(source=img_path, save=True, conf=0.25)
    # Get the most recent prediction result folder
    pred_dir = os.path.join("runs/detect", os.listdir("runs/detect")[-1])
    result_image_path = os.path.join(pred_dir, os.path.basename(img_path))
    return result_image_path

import cv2 
from datetime import timedelta
import smtplib
from email.message import EmailMessage

model = YOLO(d_data)

output_dir = "/outputs"
os.makedirs(output_dir, exist_ok=True)

def send_email_alert(vdo_name, vdo_time):
    email_sender = 'arsahadu@gmail.com'
    email_password = 'qrfs ilop gcda dksx'
    email_receiver = 'arsahadu@gmail.com'

    subject = '🔥 Fire Detected!'
    body = f'Your fire detection system has detected fire.\n\nVideo: {vdo_name}\nTime: {vdo_time}'

    msg = EmailMessage()
    msg['From'] = email_sender
    msg['To'] = email_receiver
    msg['Subject'] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_sender, email_password)
            smtp.send_message(msg)
        print("✅ Email sent successfully.")
    except Exception as e:
        print("❌ Email failed:", e)

def detect_fire_with_metrics(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_num, fire_frame_num = 0, 0
    first_fire_time, first_fire_frame = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        results = model.predict(source=frame, conf=0.25, save=False, verbose=False)
        fire_found = False

        for r in results:
            boxes = r.boxes
            if boxes and len(boxes.cls) > 0:
                fire_found = True
                for box in boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if fire_found:
            fire_frame_num += 1
            if not first_fire_time:
                first_fire_time = str(timedelta(seconds=frame_num / fps)).split('.')[0]
                first_fire_frame = frame.copy()

        out.write(frame)

    cap.release()
    out.release()

    return {
        'video_path': video_path,
        'total_frames': total_frames,
        'fire_frames': fire_frame_num,
        'fps': fps,
        'first_fire_time': first_fire_time,
        'first_fire_frame': first_fire_frame
    }


def process_videos(video1, video2):
    metrics1 = detect_fire_with_metrics(video1, f"{output_dir}/video1_annotated.mp4")
    metrics2 = detect_fire_with_metrics(video2, f"{output_dir}/video2_annotated.mp4")

    alert_sent = False
    if metrics1['first_fire_time']:
        send_email_alert(os.path.basename(metrics1['video_path']), metrics1['first_fire_time'])
        alert_sent = True
    elif metrics2['first_fire_time']:
        send_email_alert(os.path.basename(metrics2['video_path']), metrics2['first_fire_time'])
        alert_sent = True

    return metrics1['fire_frames'], metrics2['fire_frames'], alert_sent

from ultralytics import YOLO
model = YOLO(d_data)

import gradio as gr
import cv2
from PIL import Image
import numpy as np
import os
from IPython.display import Image as IPyImage
from ultralytics import YOLO

# Load your model
model = YOLO(d_data)

# Folder for output
output_dir = "/outputs"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# 🔥 1. Image Detection Handler
# ----------------------------
def detect_fire_image(image_path):
    results = model.predict(source=image_path, save=False, conf=0.25, verbose=False)
    result_img = results[0].plot()
    return Image.fromarray(result_img)

# -----------------------------
# 🔥 2. Video Detection Handler
# -----------------------------
from datetime import timedelta
import smtplib
from email.message import EmailMessage

first_fire_info = {
    'time': None,
    'video': None,
    'frame_img': None
}

def send_email_alert(vdo_name, vdo_time):
    email_sender = 'your_sender_email@gmail.com'
    email_password = 'your_app_password_here'  # Use Gmail App Password
    email_receiver = 'your_receiver_email@gmail.com'

    subject = '🔥 Fire Detected!'
    body = f'Your fire detection system has detected fire in video {vdo_name} at {vdo_time}.'

    msg = EmailMessage()
    msg['From'] = email_sender
    msg['To'] = email_receiver
    msg['Subject'] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_sender, email_password)
            smtp.send_message(msg)
        print("✅ Email sent successfully.")
    except Exception as e:
        print("❌ Email failed:", e)

def detect_fire_with_metrics(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_num = 0
    fire_detected_frames = []

    global first_fire_info
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        results = model.predict(source=frame, conf=0.25, save=False, verbose=False)
        fire_found = False

        for r in results:
            boxes = r.boxes
            if boxes and len(boxes.cls) > 0:
                fire_found = True
                for box in boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "FIRE", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if fire_found:
            fire_detected_frames.append(frame_num)
            if first_fire_info['time'] is None:
                timestamp = str(timedelta(seconds=frame_num / fps)).split('.')[0]
                first_fire_info['time'] = timestamp
                first_fire_info['video'] = os.path.basename(video_path)
                first_fire_info['frame_img'] = frame.copy()
                cv2.imwrite(f"{output_dir}/first_fire.jpg", frame)

        out.write(frame)

    cap.release()
    out.release()

    return {
        'video_path': video_path,
        'total_frames': total_frames,
        'fire_frames': fire_detected_frames,
        'fps': fps
    }

def process_videos(v1, v2):
    global first_fire_info
    first_fire_info = {
        'time': None,
        'video': None,
        'frame_img': None
    }

    videos = [v1, v2]
    metrics_list = []
    summary = ""

    for i, vid in enumerate(videos):
        output_vid = f"{output_dir}/video{i+1}_annotated.mp4"
        metrics = detect_fire_with_metrics(vid, output_vid)
        metrics_list.append(metrics)
        fire_pct = (len(metrics['fire_frames']) / metrics['total_frames']) * 100
        summary += f"🎥 Video: {os.path.basename(metrics['video_path'])}\n"
        summary += f"   🔸 Total Frames: {metrics['total_frames']}\n"
        summary += f"   🔥 Fire Frames: {len(metrics['fire_frames'])} ({fire_pct:.2f}%)\n\n"

    if first_fire_info['time']:
        send_email_alert(first_fire_info['video'], first_fire_info['time'])
        summary += f"🚨 ALERT: Fire detected in {first_fire_info['video']} at {first_fire_info['time']}\n"
        return summary, f"{output_dir}/first_fire.jpg"
    else:
        summary += "✅ No fire detected in either video.\n"
        return summary, None

# -----------------------------
# 🌟 Gradio UI (Tabbed Layout)
# -----------------------------
image_ui = gr.Interface(
    fn=detect_fire_image,
    inputs=gr.Image(type="filepath", label="Upload Fire/Smoke Image"),
    outputs=gr.Image(type="numpy", label="Annotated Image"),
    title="🔥 Fire & Smoke Detection from Image"
)

video_ui = gr.Interface(
    fn=process_videos,
    inputs=[
        gr.Video(label="Upload Video 1"),
        gr.Video(label="Upload Video 2")
    ],
    outputs=[
        gr.Textbox(label="Detection Summary"),
        gr.Image(type="filepath", label="Fiacrst Fire Frame (if detected)")
    ],
    title="🎥 Fire Detection from Videos (Email Enabled)"
)

gr.TabbedInterface(
    [image_ui, video_ui],
    ["Image Detection", "Video Detection"]
).launch(debug=True)
