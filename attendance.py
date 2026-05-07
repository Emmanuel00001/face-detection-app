import cv2
import os
import csv
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = []
labels = []
label_map = {}
current_label = 0

known_faces_dir = "known_faces"

for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    label_map[current_label] = person_name

    for image_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(current_label)

    current_label += 1

recognizer.train(faces, __import__('numpy').array(labels))
marked = set()

cap = cv2.VideoCapture(0)
print("Attendance system running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = label_map[label]
            color = (0, 255, 0)

            if name not in marked:
                marked.add(name)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("attendance.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, timestamp])
                print(f"Attendance marked for {name} at {timestamp}")
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()