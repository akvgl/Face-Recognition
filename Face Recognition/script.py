import face_recognition
import cv2
import os

known_encodings = []
known_names = []

dataset_path = "dataset"

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):
        image_path = os.path.join(person_folder, filename)

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"Loaded encoding for {person_name} from {filename}")
            else:
                print(f"No face found in {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from webcam")
        break

    face_locations = face_recognition.face_locations(frame, model="hog")
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    #for safety debug
    if len(face_encodings) != len(face_locations):
        print("Warning: Mismatch between number of encodings and locations.")
        continue
    #debug
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            print(f"Match found: {name}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) == 27: #esc to end
     break


video_capture.release()
cv2.destroyAllWindows()
