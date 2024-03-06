from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import io

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def calculate_skin_percentage(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_percentage = (cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])) * 100.0
    return skin_percentage

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    skin_percentage_threshold = 12.0  # Adjust this threshold based on your observations

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        skin_percentage = calculate_skin_percentage(frame)
        frame_count += 1

        if frame_count > 30:  # Skip initial frames for stabilization
            if len(contours) > 500 and skin_percentage > skin_percentage_threshold:
                result = 'Fake'
            else:
                result = 'Real'

            print(f'Result: {result}, Skin Percentage: {skin_percentage:.2f}%')

            if result == 'Fake':
                # Take further actions for detected deepfake, e.g., stop processing or notify
                pass

            return result

    cap.release()
    return 'Real'

def get_next_video_id():
    # Function to get the next available video ID based on existing files in the 'uploads' folder
    existing_ids = [int(file.split('_')[0]) for file in os.listdir(app.config['UPLOAD_FOLDER']) if file.endswith('.mp4')]
    return max(existing_ids, default=0) + 1

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return render_template('index.html', result='Error: No video file provided')

    video_file = request.files['video']

    if video_file.filename == '':
        return render_template('index.html', result='Error: No selected file')

    video_data = video_file.read()

    # Get the next available video ID
    video_id = get_next_video_id()

    # Save the uploaded video data to a file in the 'uploads' folder with an incrementing ID
    video_filename = f"{video_id}_temp_video.mp4"
    temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    with open(temp_video_path, 'wb') as temp_file:
        temp_file.write(video_data)

    # Perform deepfake detection using the temporary file
    result = detect_deepfake(temp_video_path)

    # Display the result immediately after uploading
    return render_template('results.html', result=result)

@app.route('/')
def index():
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)

