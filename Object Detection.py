import cv2

# Load the pre-trained Haar Cascade model for human detection
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')


# Function to detect and draw rectangles around humans in the frame
def detect_human(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect human bodies in the frame
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the detection accuracy based on the number of neighbors
        accuracy = 100 * (1 - 0.05 * humans[0][3])

        # Print the detection accuracy
        cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


# Open a connection to the camera (use 0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Call the detect_human function
    result_frame = detect_human(frame)

    # Display the result frame
    cv2.imshow('Human Detection', result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
