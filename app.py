#!/usr/bin/env python3
"""
Flask backend server for video action recognition.
Features:
- Immediate transcoding for non-MP4 video previews.
- Automatic thumbnail generation for video posters.
- Automatic cleanup of temporary files.
"""

import os
import tempfile
import json
from flask import Flask, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import subprocess
import sys
import shutil
import threading

app = Flask(__name__, static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# --- Directory Setup ---
# Holds original uploaded files temporarily before prediction.
UPLOAD_FOLDER = tempfile.mkdtemp()
# Holds transcoded MP4 videos for web preview.
PREVIEW_FOLDER = 'static/previews'
# Holds generated thumbnails for video posters.
THUMBNAIL_FOLDER = 'static/thumbnails'
os.makedirs(PREVIEW_FOLDER, exist_ok=True)
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

def schedule_deletion(file_path, delay_seconds=3600):
    """Schedules a file to be deleted after a specified delay (default 1 hour)."""
    def delete_file():
        try:
            print(f"⏲️ Deleting temporary file: {file_path}")
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

    timer = threading.Timer(delay_seconds, delete_file)
    timer.start()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_thumbnail(video_path, output_filename):
    """Extracts a frame from a video to use as a thumbnail poster."""
    output_path = os.path.join(THUMBNAIL_FOLDER, output_filename)
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', '00:00:01.000',
            '-vframes', '1',
            '-q:v', '2',
            '-y', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        thumbnail_url = url_for('static', filename=f'thumbnails/{output_filename}', _external=True)
        return {"url": thumbnail_url, "path": output_path}
    except Exception as e:
        print(f"Thumbnail generation error: {e}")
        return None

def transcode_for_preview(input_path, output_filename):
    """Converts a video to a web-friendly MP4 format."""
    output_path = os.path.join(PREVIEW_FOLDER, output_filename)
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
        preview_url = url_for('static', filename=f'previews/{output_filename}', _external=True)
        return {"url": preview_url, "path": output_path}
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return None

def predict_video_action(video_path):
    """Runs the external prediction script on a video file."""
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
            temp_output = temp_file.name
        
        cmd = [
            sys.executable, 'predict_single_video.py', 
            '--video', video_path,
            '--output', temp_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {"error": f"Prediction script failed: {result.stderr}"}
        
        with open(temp_output, 'r') as f:
            prediction_data = json.load(f)
        
        os.unlink(temp_output)
        return prediction_data
        
    except Exception as e:
        return {"error": f"Prediction process error: {str(e)}"}

@app.route('/')
def index():
    """Serves the main HTML page."""
    return send_from_directory('.', 'video_predictor.html')

@app.route('/generate_preview', methods=['POST'])
def generate_preview():
    """Handles file upload, generates a preview and thumbnail, and schedules their deletion."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type or no file selected"}), 400

    filename = secure_filename(file.filename)
    original_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(original_path)
    
    base_name = os.path.splitext(filename)[0]
    preview_filename = f"{base_name}_preview.mp4"
    thumbnail_filename = f"{base_name}_thumb.jpg"

    transcode_result = transcode_for_preview(original_path, preview_filename)
    thumbnail_result = generate_thumbnail(original_path, thumbnail_filename)
    
    if transcode_result and thumbnail_result:
        # Schedule both temporary files for deletion after 1 hour
        schedule_deletion(transcode_result['path'], 3600)
        schedule_deletion(thumbnail_result['path'], 3600)
        
        return jsonify({
            "preview_url": transcode_result['url'],
            "poster_url": thumbnail_result['url'],
            "original_filename": filename
        })
    else:
        return jsonify({"error": "Failed to generate preview or thumbnail"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Runs prediction on a file that has already been uploaded."""
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({"error": "Filename not provided"}), 400
    
    filename = data['filename']
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(video_path):
        return jsonify({"error": "File not found on server. It may have been cleared or never uploaded."}), 404
    
    prediction_result = predict_video_action(video_path)
    
    if "error" in prediction_result:
        return jsonify(prediction_result), 500
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    print("Starting Video Action Recognition Server...")
    print("Open your browser and go to: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)