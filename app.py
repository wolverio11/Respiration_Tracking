"""
app.py
======
Flask web server that serves the dashboard and streams camera data.

Routes:
    GET  /            -> Dashboard HTML page
    GET  /video_feed  -> Live MJPEG camera stream
    GET  /data        -> JSON state (polled every second by browser)
    POST /reset       -> Reset session and recalibrate

Run with:   python app.py
Then open:  http://localhost:5000
"""

import sys
import io
import time
from flask import Flask, Response, jsonify, request, send_from_directory
from session_runner import SessionRunner

app = Flask(__name__, static_folder="static")

# Create and start the pipeline
pipeline = SessionRunner(camera_index=0)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/video_feed")
def video_feed():
    """
    MJPEG live camera stream.
    The browser treats this as a continuously updating image.
    Each frame is sent as a JPEG inside a multipart HTTP response.
    """
    def generate():
        while True:
            frame_bytes = pipeline.get_frame()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            time.sleep(1.0 / 30)   # 30 fps

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/data")
def data():
    """Returns current system state as JSON. Polled every 1 second."""
    return jsonify(pipeline.get_state())


@app.route("/reset", methods=["POST"])
def reset():
    """Resets the session -- recalibrates from scratch."""
    pipeline.reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Fix for Windows terminals using cp1252 encoding which cannot
    # display certain Unicode characters. This wraps stdout with
    # utf-8 and replaces any unencodable characters safely.
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    print("\n" + "=" * 55)
    print("  Pranayama PLIM Monitor  (Lee et al. 2021)")
    print("  Open browser at: http://localhost:5000")
    print("  Breathe normally for 10 seconds to calibrate.")
    print("=" * 55 + "\n")

    pipeline.start()
    try:
        app.run(host="0.0.0.0", port=5000,
                debug=False, threaded=True)
    finally:
        pipeline.stop()
