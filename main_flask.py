
#image classification trashbot code


from flask import Flask, render_template, request, Response, jsonify
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

app = Flask(__name__)

# Paramètres globaux
model = None
labels = []
arduino = None
cap = None
ARDUINO_PORT = "COM12"
CLASS_CONFIDENCE = 0.7 #change  this value if u want

# Fonction pour initialiser la base de données
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

# Fonction pour nettoyer les noms de classe
def clean_class_name(class_name):
    # Extrait le type de matériau (plastic, metal, glass, etc.) du label
    if class_name and isinstance(class_name, str):
        match = re.search(r'^\d+\s+(\w+)$', class_name.strip())
        if match:
            return match.group(1).lower()  # Retourne le nom en minuscules
        else:
            return class_name.lower()
    return "unknown"

# Fonction pour sauvegarder les données dans la base de données
def save_to_db(material_class, confidence, image_path):
    try:
        # Nettoyer le nom de la classe avant de l'enregistrer
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
        print(f"✅ Données sauvegardées en DB: {cleaned_class_name}, {confidence}")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde en DB: {e}")
        return False

# Fonction pour récupérer l'historique des classifications
def get_history(limit=50):
    try:
        conn = sqlite3.connect('waste_sorting.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM sorting_history ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()

        # Nettoyer les noms de classes
        cleaned_rows = []
        for row in rows:
            material_class = clean_class_name(row['material_class'])
            cleaned_row = {key: (material_class if key == 'material_class' else row[key]) for key in row.keys()}
            cleaned_rows.append(cleaned_row)

        return cleaned_rows

    except Exception as e:
        print(f"❌ Erreur lors de la récupération de l'historique: {e}")
        return []
    
# Fonction pour obtenir le nom de classe complet
def get_full_class_name(class_index):
    if 0 <= class_index < len(labels):
        return labels[class_index].strip()
    return "Unknown"



def initialize_system():
    global model, labels, arduino, cap

    # Création du dossier pour les images capturées s'il n'existe pas
    os.makedirs('static/captured_images', exist_ok=True)

    # 1) Chargement du modèle TensorFlow
    try:
        model = load_model("converted_keras/keras_model.h5")
        print("✅ Modèle chargé avec succès!")
    except Exception as e:
        print(f"⚠️ Erreur de chargement du modèle: {e}")
        return False

    # 2) Chargement des étiquettes
    try:
        with open("converted_keras/labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"✅ Étiquettes chargées avec succès! Labels: {labels}")
    except Exception as e:
        print(f"⚠️ Erreur de chargement des étiquettes: {e}")
        return False

    # 3) Initialisation de la caméra
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️ Erreur: Impossible d'ouvrir la caméra.")
            return False
        print("✅ Caméra initialisée avec succès!")
    except Exception as e:
        print(f"⚠️ Erreur d'initialisation de la caméra: {e}")
        return False

    # 4) Initialisation de l'Arduino
    try:
        if arduino is not None and arduino.is_open:
            arduino.close()
            print("🔄 Port série déjà ouvert, fermeture propre avant reconnexion.")

        arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
        time.sleep(2)  # Laisser le temps à l'Arduino de s'initialiser

        print("✅ Arduino connecté!")

        # Envoyer la commande RESET et attendre la stabilisation
        arduino.write("RESET\n".encode())
        print("⚙️ Moteur stepper réinitialisé.")
        time.sleep(5)
        print("✅ Moteur stepper prêt!")

    except serial.SerialException as e:
        print(f"⚠️ Erreur de connexion à l'Arduino: {e}")
        arduino = None

    return True

# Mapping des étiquettes de classe aux commandes de bac
bin_commands = {
    "plastic": "CW",         # Bac 1
    "metal": "CW CW",      # Bac 2
    "glass": "CW CW CW",   # Bac 3
    "unknown": "CW CW CW CW" # Bac 4 (360°)
}

# Fonction pour obtenir la commande de bac pour une classe donnée
def get_bin_command(class_name):
    # Nettoie le nom de classe et recherche la commande correspondante
    clean_name = clean_class_name(class_name)
    command = bin_commands.get(clean_name)
    
    if command is None:
        print(f"⚠️ Aucune commande trouvée pour la classe '{class_name}', utilisation de la commande 'unknown'")
        return bin_commands["unknown"]
    
    return command

# Fonction pour envoyer une commande au moteur
def send_motor_command(command):
    if arduino and arduino.is_open:
        try:
            for cmd in command.split():
                arduino.write(f"{cmd}\n".encode())
                print(f"🛠️ Envoyé: {cmd}")
                time.sleep(2.5)  # Attendre pour chaque rotation
            arduino.write("STOP\n".encode())
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi de la commande au moteur: {e}")
            return False
    else:
        print("⚠️ Erreur: Arduino non connecté.")
        return False

# Fonction pour capturer une image depuis la webcam
def capture_image():
    if cap is None or not cap.isOpened():
        print("❌ Erreur: Caméra non disponible")
        return None
    
    try:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Erreur: Impossible de lire depuis la caméra.")
            return None
        
        return frame
    except Exception as e:
        print(f"❌ Erreur lors de la capture d'image: {e}")
        return None

# Fonction pour classifier une image
def classify_image(frame):
    if model is None:
        print("❌ Erreur: Modèle non chargé")
        return "unknown", 0.0, None
    
    try:
        # Générer un nom de fichier unique basé sur l'horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"static/captured_images/captured_{timestamp}.jpg"
        
        # Sauvegarder l'image
        cv2.imwrite(img_path, frame)
        
        # Prétraiter l'image pour le modèle
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        # Faire une prédiction
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        confidence = float(predictions[0][class_index])
        
        # Obtenir le nom complet de la classe
        full_class_name = get_full_class_name(class_index)
        
        print(f"🔍 Classification: index={class_index}, nom={full_class_name}, confiance={confidence:.4f}")
        
        return full_class_name, confidence, img_path
    except Exception as e:
        print(f"❌ Erreur lors de la classification: {e}")
        return "unknown", 0.0, None

# Fonction pour générer le flux vidéo
def gen_frames():
    while True:
        if cap is None or not cap.isOpened():
            break
        
        try:
            success, frame = cap.read()
            if not success:
                break
            
            # Convertir en JPEG pour le streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"❌ Erreur dans gen_frames: {e}")
            break

# Routes Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classify', methods=['POST'])
def classify():
    print("🚨 Requête de classification reçue.")
    frame = capture_image()
    if frame is None:
        print("❌ Erreur capture image")
        return jsonify({'error': 'Failed to capture image'}), 400

    # Classifier l'image
    class_name, confidence, img_path = classify_image(frame)
    print(f"🔔 Résultat classification: classe={class_name}, confiance={confidence}, chemin={img_path}")

    # Vérifier la confiance
    if confidence < CLASS_CONFIDENCE:  # Seuil de confiance
        print("⚠️ Confiance faible, mais l'action sera exécutée.")

    # Stockage en base de données
    save_to_db(class_name, confidence, img_path)

    # Commande moteur basée sur la classification
    motor_command = bin_commands.get(class_name, "CW CW CW CW")
    print(f"🛠️ Commande moteur envoyée: {motor_command}")
    motor_success = send_motor_command(motor_command)

    return jsonify({
        'class': class_name,
        'confidence': f"{confidence:.2f}",
        'image_path': img_path,
        'motor_command': motor_command,
        'motor_success': motor_success,
        'low_confidence': confidence < CLASS_CONFIDENCE  # Indicateur de faible confiance
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

if __name__ == '__main__':
    init_db()
    initialize_system()
    app.run(debug=True, use_reloader=False)