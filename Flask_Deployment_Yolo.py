from flask import Flask, Response, render_template_string
import cv2
from ultralytics import YOLO

# --- Load model ---
model = YOLO("screw_detector_retrain/from_best_hyp/weights/best.pt")
CONFIDENCE_THRESHOLD = 0.5

# --- Setup camera ---
cap = cv2.VideoCapture(0)

app = Flask(__name__)

# --- Web interface HTML ---
html_template = """
<!doctype html>
<title>Screw Detection</title>
<h2 style="text-align:center;">Live Screw Detection Feed</h2>
<img src="{{ url_for('video_feed') }}" width="720">
"""

# --- Video streaming generator ---
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run inference
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        annotated_frame = results.plot()

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Routes ---
@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Start server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
