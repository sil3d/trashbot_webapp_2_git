import cv2
import numpy as np
import serial
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# 1) Load the TensorFlow model
model = load_model("converted_keras/keras_model.h5")

# 2) Load labels
with open("converted_keras/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 3) Initialize camera (Important!)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ Error: Could not open the camera.")
    # You might want to exit or handle the error here:
    exit(1)

# 4) Open serial connection to Arduino
try:
    arduino = serial.Serial('COM12', 9600, timeout=1)
    time.sleep(2)  # Allow Arduino to initialize
    print("✅ Arduino connected!")
    arduino.write("RESET\n".encode())  # Home the stepper
    time.sleep(5)
    print("✅ Stepper motor ready!")
except Exception as e:
    print("⚠️ Error: Could not connect to Arduino.")
    print(e)
    arduino = None

# 5) Mapping class labels to bin movements 
# (Still using time-based “CW” commands, but at least you won't crash now.)
bin_commands = {
    "plastic": "CW",         # Bin 1
    "metal":   "CW CW",      # Bin 2
    "glass":   "CW CW CW",   # Bin 3
    "unknown": "CW CW CW CW" # Bin 4 (360°)
}

def classify_image():
    """Captures an image from the webcam, classifies it, and returns the predicted label."""
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error: Cannot read from camera.")
        return "unknown"

    # Save the frame to a file
    img_path = "captured_image.jpg"
    cv2.imwrite(img_path, frame)

    # Preprocess the image for the model
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make a prediction
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    return labels[class_index]

def send_motor_command(command):
    """Sends motor command to Arduino via serial, splitting on spaces."""
    if arduino and arduino.is_open:
        for cmd in command.split():
            arduino.write(f"{cmd}\n".encode())
            print(f"🛠️ Sent: {cmd}")
            time.sleep(2.5)  # Wait for each rotation
        arduino.write("STOP\n".encode())
    else:
        print("⚠️ Error: Arduino not connected.")

# 6) Main loop
while True:
    print("\n🔵 Choose an option:")
    print("1️⃣ Automatic AI classification")
    print("2️⃣ Manual motor test")
    print("3️⃣ Exit")
    option = input("\nEnter option (1, 2, or 3): ").strip()

    if option == "1":
        input("\n🔵 Press Enter to classify an object...")
        classified_label = classify_image()
        print("\n" + "=" * 40)
        print(f"✅ DETECTED OBJECT: {classified_label.upper()}")
        print("=" * 40 + "\n")

        motor_command = bin_commands.get(classified_label, "CW CW CW CW")  # Default to last bin if not found
        send_motor_command(motor_command)
    elif option == "2":
        manual_class = input("\nEnter class (plastic, metal, glass, unknown): ").strip().lower()
        if manual_class in bin_commands:
            send_motor_command(bin_commands[manual_class])
        else:
            print("⚠️ Invalid class!")
    elif option == "3":
        print("🚪 Exiting program.")
        break
    else:
        print("⚠️ Invalid option! Enter 1, 2, or 3.")

# Optionally, when finished, release the camera:
cap.release()
