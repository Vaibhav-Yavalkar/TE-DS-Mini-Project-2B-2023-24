from flask import Flask, render_template, request
import os
import pytesseract  # Import pytesseract here

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\HP\OneDrive\Desktop\sem6\Tesseract-OCR\tesseract.exe'

from deeplearning import object_detection  # Import your deep learning module

# web server gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, r'C:\Users\HP\OneDrive\Desktop\sem6\NumberPlate-Detection-Extraction-main\static\upload')


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)

        if filename.endswith('.mp4') or filename.endswith('.avi'):  # Video file
            # Provide the video filename directly for demonstration purposes
            video_filename = 'video2.mp4'
            video_path = os.path.join(UPLOAD_PATH, video_filename)
            text_list, processed_frames = object_detection(video_path, video_filename)
            return render_template('index.html', upload=True, upload_video=video_filename, text=text_list,
                                   frames=processed_frames)
        else:
            text_list, _ = object_detection(path_save, filename)
            return render_template('index.html', upload=True, upload_image=filename, text=text_list, no=len(text_list))

    return render_template('index.html', upload=False)

if __name__ == "__main__":
    app.run(debug=True)
