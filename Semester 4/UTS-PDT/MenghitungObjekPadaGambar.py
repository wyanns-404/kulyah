import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video stream (0 for webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Check if the frame is read correctly
    if not ret:
        break
    
    # Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Count the number of faces detected
    num_faces = len(faces)
    
    # Display the number of faces in the top left corner
    cv2.putText(frame, f'Faces: {num_faces}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the frame with bounding boxes and face count
    cv2.imshow('Face Detector', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
