from flask import Flask, render_template, request, Response, jsonify, send_file
import cv2
import numpy as np
import serial
import time
import os
from datetime import datetime
import sqlite3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import re
import pandas as pd


app = Flask(__name__)

# Global parameters
model = None
labels = []
arduino = None
cap = None
ARDUINO_PORT = "COM12"
CLASS_CONFIDENCE = 0.7  # Change this value if needed

# Function to initialize the database
def init_db():
    conn = sqlite3.connect('waste_sorting.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sorting_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp DATETIME NOT NULL,
        image_path TEXT
    )
    ''')
    conn.commit()
    conn.close()

# Function to clean class names
def clean_class_name(class_name):
    # Extract the material type (plastic, metal, glass, etc.) from the label
    if class_name and isinstance(class_name, str):
        match = re.search(r'^\d+\s+(\w+)$', class_name.strip())
        if match:
            return match.group(1).lower()  # Return lowercase class name
        else:
            return class_name.lower()
    return "unknown"

# Function to save data to the database
def save_to_db(material_class, confidence, image_path):
    try:
        cleaned_class_name = clean_class_name(material_class)

        conn = sqlite3.connect('waste_sorting.db')
        cursor = conn.cursor()
        timestamp = datetime.now()
        cursor.execute(
            'INSERT INTO sorting_history (material_class, confidence, timestamp, image_path) VALUES (?, ?, ?, ?)',
            (cleaned_class_name, confidence, timestamp, image_path)
        )
        conn.commit()
        conn.close()
        print(f"✅ Data saved to DB: {cleaned_class_name}, {confidence}")
        return True
    except Exception as e:
        print(f"❌ Error saving to DB: {e}")
        return False

# Function to retrieve classification history
def get_history(limit=50):
    try:
        conn = sqlite3.connect('waste_sorting.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM sorting_history ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()

        # Clean class names
        cleaned_rows = []
        for row in rows:
            material_class = clean_class_name(row['material_class'])
            cleaned_row = {key: (material_class if key == 'material_class' else row[key]) for key in row.keys()}
            cleaned_rows.append(cleaned_row)

        return cleaned_rows

    except Exception as e:
        print(f"❌ Error retrieving history: {e}")
        return []

# Function to get the full class name from index
def get_full_class_name(class_index):
    if 0 <= class_index < len(labels):
        return labels[class_index].strip()
    return "Unknown"

# Function to initialize the system
def initialize_system():
    global model, labels, arduino, cap

    # Create folder for captured images if not present
    os.makedirs('static/captured_images', exist_ok=True)

    # 1) Load TensorFlow model
    try:
        model = load_model("converted_keras/keras_model.h5")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return False

    # 2) Load labels
    try:
        with open("converted_keras/labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"✅ Labels loaded successfully! Labels: {labels}")
    except Exception as e:
        print(f"⚠️ Error loading labels: {e}")
        return False

    # 3) Initialize the camera
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️ Error: Could not open camera.")
            return False
        print("✅ Camera initialized successfully!")
    except Exception as e:
        print(f"⚠️ Error initializing camera: {e}")
        return False

    # 4) Initialize Arduino
    try:
        if arduino is not None and arduino.is_open:
            arduino.close()
            print("🔄 Serial port already open, closing properly before reconnecting.")

        arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
        time.sleep(2)  # Let the Arduino initialize

        print("✅ Arduino connected!")

        # Send RESET command and wait for stabilization
        arduino.write("RESET\n".encode())
        print("⚙️ Stepper motor reset.")
        time.sleep(5)
        print("✅ Stepper motor ready!")

    except serial.SerialException as e:
        print(f"⚠️ Arduino connection error: {e}")
        arduino = None

    return True

# Map class labels to bin motor commands
bin_commands = {
    "plastic": "CW",              # Bin 1
    "metal": "CW CW",             # Bin 2
    "glass": "CW CW CW",          # Bin 3
    "unknown": "CW CW CW CW"      # Bin 4 (360° rotation)
}

# Get motor command for a given class
def get_bin_command(class_name):
    clean_name = clean_class_name(class_name)
    command = bin_commands.get(clean_name)
    
    if command is None:
        print(f"⚠️ No command found for class '{class_name}', using 'unknown'")
        return bin_commands["unknown"]
    
    return command

# Send motor command to Arduino
def send_motor_command(command):
    if arduino and arduino.is_open:
        try:
            for cmd in command.split():
                arduino.write(f"{cmd}\n".encode())
                print(f"🛠️ Sent: {cmd}")
                time.sleep(2.5)  # Wait between rotations
            arduino.write("STOP\n".encode())
            return True
        except Exception as e:
            print(f"❌ Error sending motor command: {e}")
            return False
    else:
        print("⚠️ Error: Arduino not connected.")
        return False

# Capture image from webcam
def capture_image():
    if cap is None or not cap.isOpened():
        print("❌ Error: Camera not available")
        return None
    
    try:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Error: Unable to read from camera.")
            return None
        
        return frame
    except Exception as e:
        print(f"❌ Error capturing image: {e}")
        return None

# Classify an image using the loaded model
def classify_image(frame):
    if model is None:
        print("❌ Error: Model not loaded")
        return "unknown", 0.0, None
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"static/captured_images/captured_{timestamp}.jpg"
        
        cv2.imwrite(img_path, frame)
        
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        confidence = float(predictions[0][class_index])
        
        full_class_name = get_full_class_name(class_index)
        
        print(f"🔍 Classification: index={class_index}, name={full_class_name}, confidence={confidence:.4f}")
        
        return full_class_name, confidence, img_path
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return "unknown", 0.0, None

# Generate video stream
def gen_frames():
    while True:
        if cap is None or not cap.isOpened():
            break
        
        try:
            success, frame = cap.read()
            if not success:
                break
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"❌ Error in gen_frames: {e}")
            break

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classify', methods=['POST'])
def classify():
    print("🚨 Classification request received.")
    frame = capture_image()
    if frame is None:
        print("❌ Image capture error")
        return jsonify({'error': 'Failed to capture image'}), 400

    class_name, confidence, img_path = classify_image(frame)
    print(f"🔔 Classification result: class={class_name}, confidence={confidence}, path={img_path}")

    if confidence < CLASS_CONFIDENCE:
        print("⚠️ Low confidence, but action will still be taken.")

    save_to_db(class_name, confidence, img_path)

    motor_command = bin_commands.get(class_name, "CW CW CW CW")
    print(f"🛠️ Motor command sent: {motor_command}")
    motor_success = send_motor_command(motor_command)

    return jsonify({
        'class': class_name,
        'confidence': f"{confidence:.2f}",
        'image_path': img_path,
        'motor_command': motor_command,
        'motor_success': motor_success,
        'low_confidence': confidence < CLASS_CONFIDENCE
    })

@app.route('/history')
def history():
    data = get_history()
    return render_template('history.html', data=data)

@app.route('/manual_control', methods=['POST'])
def manual_control():
    class_name = request.form.get('class', 'unknown')
    
    motor_command = get_bin_command(class_name)
    motor_success = send_motor_command(motor_command)
    
    return jsonify({
        'class': class_name,
        'motor_command': motor_command,
        'motor_success': motor_success
    })

@app.route('/status')
def status():
    system_status = {
        'model_loaded': model is not None,
        'camera_connected': cap is not None and cap.isOpened(),
        'arduino_connected': arduino is not None and arduino.is_open,
        'labels': labels if labels else []
    }
    return jsonify(system_status)

@app.route('/download_report')
def download_report():
    # Connect to the database
    conn = sqlite3.connect('waste_sorting.db')

    # Read the data into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM history", conn)

    # Close the connection
    conn.close()

    # Define the path for the Excel report
    report_path = os.path.join('static', 'waste_report.xlsx')

    # Export DataFrame to Excel
    df.to_excel(report_path, index=False)

    # Send the Excel file to the user
    return send_file(report_path, as_attachment=True)



if __name__ == '__main__':
    init_db()
    initialize_system()
    app.run(debug=True, use_reloader=False)
