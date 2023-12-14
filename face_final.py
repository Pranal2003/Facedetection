import cv2
import face_recognition

# Load known face images and encode them
known_face_images = [
    face_recognition.load_image_file("Pranal.jpg"),  
    face_recognition.load_image_file("Mayur.jpg"), 
    # Add more known face images as needed
]

known_face_encodings = [face_recognition.face_encodings(img)[0] for img in known_face_images]

# Create an array of known face encodings and corresponding names
known_face_names = ["Pranal", "Mayur"]  # Add names corresponding to the known face images

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and display the name
        top, right, bottom, left = face_recognition.face_locations(frame)[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()
