import numpy as np
import cv2
import pytesseract as pt

pt.pytesseract.tesseract_cmd = r'C:\Users\HP\OneDrive\Desktop\sem6\Tesseract-OCR\tesseract.exe' 
# LOAD YOLO MODEL
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX(r'C:\Users\HP\OneDrive\Desktop\sem6\NumberPlate-Detection-Extraction-main\static\models\best.onnx')

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def get_detections(img, net):
    # CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections


def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    # NMS
    index = np.array(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)).flatten()

    return boxes_np, confidences_np, index


def extract_text(image, bbox):
    x, y, w, h = bbox

    roi = image[y:y+h, x:x+w]
    if 0 in roi.shape:
        return ''
    else:
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        
        # Apply additional preprocessing (e.g., contrast adjustment, sharpening) to enhance text extraction
        enhanced_roi = apply_image_enhancement(roi_bgr)
        
        # Perform OCR on the enhanced ROI
        text = pt.image_to_string(enhanced_roi, lang='eng', config='--psm 6')
        text = text.strip()

        return text


def apply_image_enhancement(image):
    # Apply contrast adjustment
    enhanced_img = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    
    # Apply sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_img = cv2.filter2D(enhanced_img, -1, kernel)

    return sharpened_img


def drawings(image, boxes_np, confidences_np, index):
    # drawings
    text_list = []
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = extract_text(image, boxes_np[ind])

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 30), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        text_list.append(license_text)

    return image, text_list


# predictions
def yolo_predictions(img, net):
    ## step-1: detections
    input_image, detections = get_detections(img, net)
    ## step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    ## step-3: Drawings
    result_img, text = drawings(img, boxes_np, confidences_np, index)
    return result_img, text


def object_detection(input_path, filename):
    if input_path.endswith('.mp4') or input_path.endswith('.avi'):  # Video file
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties for the output video settings
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define the codec and create VideoWriter object for the output video
        out = cv2.VideoWriter('./static/predict/processed_{}'.format(filename),
                              cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, 
                              (frame_width, frame_height))
        
        text_list = []  # To store detected texts from all frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if no frame is captured/ end of video

            result_img, text = yolo_predictions(frame, net)  # Process each frame
            text_list.extend(text)  # Collect text results
            
            out.write(result_img)  # Write the processed frame to the output video

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Return detected texts and path to the processed video
        return text_list, './static/predict/processed_{}'.format(filename)
    else:  # Image file
        image = cv2.imread(input_path)
        result_img, text_list = yolo_predictions(image, net)  # Process image
        saved_path = './static/predict/{}'.format(filename)  # Define save path
        cv2.imwrite(saved_path, result_img)  # Save processed image
        return text_list, saved_path 


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img
        return buf
