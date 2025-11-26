import cv2
import csv
import os
from datetime import datetime

CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "face_model.yml"
USER_NAME_FILE = "user_name.txt"
CHECKINS_FILE = "checkins.csv"

SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (80, 80)
CONFIDENCE_LIMIT = 70.0

def load_user_name():
    if not os.path.exists(USER_NAME_FILE):
        return None
    with open(USER_NAME_FILE, "r", encoding="utf-8") as f:
        return f.read().strip() or None

def append_checkin(user_name, habit_name):
    new_file = not os.path.exists(CHECKINS_FILE)
    with open(CHECKINS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "user_name", "habit"])
        ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        writer.writerow([ts, user_name, habit_name])
    print(f"\n✔ Check-in registrado: {user_name} - {habit_name} ({ts})")

def main():
    user_name = load_user_name()
    if not user_name:
        print("Nenhum usuário cadastrado. Rode antes: python setup_user.py")
        return

    if not os.path.exists(MODEL_PATH):
        print("Modelo não encontrado. Rode antes: python setup_user.py")
        return

    habit_name = input("Hábito a registrar (ex: Beber agua): ").strip()
    if not habit_name:
        habit_name = "Habito saudavel"

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("Erro no OpenCV. Instale: pip install opencv-contrib-python")
        return

    recognizer.read(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não consegui acessar a câmera.")
        return

    print("\nJanela aberta.")
    print("Use 'c' para registrar o check-in e 'q' para sair.\n")

    last_is_user = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )

        label_text = ""
        color = (0, 0, 255)
        last_is_user = False

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face)

            if confidence < CONFIDENCE_LIMIT:
                label_text = f"{user_name} ({confidence:.1f})"
                color = (0, 255, 0)
                last_is_user = True
            else:
                label_text = f"Desconhecido ({confidence:.1f})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        info_text = f"Hábito: {habit_name} | 'c' = check-in | 'q' = sair"
        cv2.putText(frame, info_text,
                    (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow("Check-in de Habitos Saudaveis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if last_is_user:
                append_checkin(user_name, habit_name)
            else:
                print("\nNão reconheci ninguém no momento do check-in.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
