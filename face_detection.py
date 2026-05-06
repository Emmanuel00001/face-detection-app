import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')
gender_net = cv2.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt')
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']
cap = cv2.VideoCapture(0)
print("Camera started. Press Q to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_net.setInput(blob)
        gender_pred = gender_net.forward()
        gender = gender_labels[gender_pred[0].argmax()]
        age_net.setInput(blob)
        age_pred = age_net.forward()
        age = age_labels[age_pred[0].argmax()]
        label = f'{gender}, {age}'
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Faces: {len(faces)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  
    cv2.imshow('Face Detection - Kay', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()