import cv2
import numpy as np

CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "face_model.yml"
USER_NAME_FILE = "user_name.txt"

SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (80, 80)
NUM_IMAGES = 30

def main():
    name = input("Seu nome: ").strip()
    if not name:
        print("Nome inválido.")
        return

    with open(USER_NAME_FILE, "w", encoding="utf-8") as f:
        f.write(name)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não consegui abrir a câmera.")
        return

    print("\nOk, olhando para a câmera...")
    print(f"Vou capturar {NUM_IMAGES} imagens suas. Aperte 'q' se quiser parar.\n")

    faces = []
    labels = []
    collected = 0
    label = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )

        for (x, y, w, h) in detected:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            faces.append(face)
            labels.append(label)
            collected += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{collected}/{NUM_IMAGES}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

            if collected >= NUM_IMAGES:
                break

        cv2.imshow("Cadastro do Usuario", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cancelado.")
            break

        if collected >= NUM_IMAGES:
            print("Coleta finalizada.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected == 0:
        print("Nenhuma imagem capturada.")
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
    except:
        print("Erro no OpenCV. Instale: pip install opencv-contrib-python")
        return

    faces_np = np.array(faces)
    labels_np = np.array(labels)

    print("Treinando o modelo...")
    recognizer.train(faces_np, labels_np)
    recognizer.save(MODEL_PATH)

    print(f"Modelo salvo em {MODEL_PATH}")
    print(f"Nome salvo em {USER_NAME_FILE}")

if __name__ == "__main__":
    main()
