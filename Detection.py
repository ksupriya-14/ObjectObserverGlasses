
from datetime import datetime
import cv2
from ultralytics import YOLO
import mediapipe as mp
import cvzone
import math
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(3, 1980)
cap.set(4, 1080)

# YOLO Models
model = YOLO("Gesture_Model.pt")
model_object = YOLO("Weights/yolov8x.pt")

# ClassNames for objects
classNames_thumbs = ['Okh_Gesture', 'ThumbsDown_Gesture', 'ThumbsUp_Gesture']

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for device state and confirmation
action_status = False
confirmation_received = False
Device_Status = False

fan_speed = 0  # Range from 0 to 110
fan_speed_max = 110
fan_speed_min = 0  # Set a minimum fan speed (0%) to start from zero

# Initialize your desired fan speed intervals
fan_speed_interval = 5  # Change the interval to 5%

# Bar parameters
bar_width = 20
bar_height = 200
bar_position = (50, 50)

# Initialize previous fan speed variable
prev_fan_speed = 0
speed_bright =0

# Initialize smoothing parameters
smoothing_factor = 0.5
prev_avg_distance = 0

# Device Selection Variables
finger_was_outside = True
selected_object = None
finger_in_box = False
mp_hands_N = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
device_selected = False
Device_Connected = False

def object_detection(img):
    global selected_object
    global finger_in_box
    global finger_was_outside
    global device_selected

    # Run YOLO model for object detection
    object_results = model_object(img, stream=True)
    start_time = datetime.now()

    # Reset selected_object when no objects are detected
    selected_object = None

    for object_result in object_results:
        boxes = object_result.boxes
        for box in boxes:
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])

            # Get the class name from classNames list
            if classNames[cls] == "cell phone" and conf >= 0.5:
                selected_object = box
                break  # Found object, no need to continue searching

        if selected_object is not None:
            end_time = datetime.now()
            detection_time = (end_time - start_time).total_seconds()

            # Hand tracking using MediaPipe
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results_hands = mp_hands_N.process(frame_rgb)

            # If hands are detected, draw landmarks
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    # Check if finger is in the box around the cell phone
                    if selected_object is not None:
                        x, y, w, h = selected_object.xyxy[0].cpu().numpy().astype(int)
                        finger_x, finger_y = int(hand_landmarks.landmark[8].x * img.shape[1]), int(
                            hand_landmarks.landmark[8].y * img.shape[0])

                        if x < finger_x < w and y < finger_y < h:
                            if finger_was_outside:  # Only change the state if the finger was previously outside the box
                                finger_in_box = not finger_in_box
                                finger_was_outside = False
                        else:
                            finger_was_outside = True  # Set the flag to True when the finger is outside the box

            # Draw bounding box around the cell phone and change color based on finger position
            if finger_in_box:
                device_selected = True
            else:
                device_selected = False

            x, y, w, h = selected_object.xyxy[0].cpu().numpy().astype(int)
            color = (0, 255, 0) if device_selected else (0, 0, 255)
            cv2.rectangle(img, (x, y), (w, h), color, 2)
    device_text = f"Device selected: {'Yes' if device_selected else 'No'}"
    cv2.putText(img, device_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
def Gesture_detection(img):
    global action_status
    # Run YOLO model for object detection
    yolo_results = model(img, stream=True)
    for yolo_result in yolo_results:
        boxes = yolo_result.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box on image
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Extract confidence score and class index
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])

            if (classNames_thumbs[cls] == 'ThumbsDown_Gesture') and (0.5<=conf):
                action_status = False
                cvzone.putTextRect(img, f"{classNames_thumbs[cls]}{conf}", (max(0, x1), max(35, y1)), scale=0.7,
                                   thickness=1)
            elif (classNames_thumbs[cls] == 'ThumbsUp_Gesture') and (0.5<=conf):
                action_status = True
                cvzone.putTextRect(img, f"{classNames_thumbs[cls]}{conf}", (max(0, x1), max(35, y1)), scale=0.7,
                                   thickness=1)
def mediapipie_detection(frame):
    global confirmation_received
    global Device_Status
    global okh_detected
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results2 = hands.process(rgb_frame)

    # Initialize confirmation flag
    okh_detected = False

    # Check for multi-hand landmarks
    if results2.multi_hand_landmarks:
        for hand_landmarks in results2.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Draw landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate distances for gesture detection
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip_x, thumb_tip_y = thumb_tip.x, thumb_tip.y
            index_finger_tip_x, index_finger_tip_y = index_finger_tip.x, index_finger_tip.y

            distance_1 = math.sqrt(
                (thumb_tip_x - index_finger_tip_x) ** 2 + (thumb_tip_y - index_finger_tip_y) ** 2)

            middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

            middle_finger_tip_x, middle_finger_tip_y = middle_finger_tip.x, middle_finger_tip.y
            ring_finger_tip_x, ring_finger_tip_y = ring_finger_tip.x, ring_finger_tip.y
            pinky_tip_x, pinky_tip_y = pinky_tip.x, pinky_tip.y

            distance_2 = max(
                math.sqrt((thumb_tip_x - middle_finger_tip_x) ** 2 + (thumb_tip_y - middle_finger_tip_y) ** 2),
                math.sqrt((thumb_tip_x - ring_finger_tip_x) ** 2 + (thumb_tip_y - ring_finger_tip_y) ** 2),
                math.sqrt((thumb_tip_x - pinky_tip_x) ** 2 + (thumb_tip_y - pinky_tip_y) ** 2))

            # Check if both conditions are met for gesture detection
            if distance_1 < 0.05 and distance_2 > 0.2:
                okh_detected = True
def Speed_Controlling(frame):
    global prev_fan_speed
    global prev_avg_distance
    global speed_bright
    global Device_Connected
    global new_fan_speed

    # Convert frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand landmarks
            landmarks = hand_landmarks.landmark

            # Get distances between finger landmarks
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate distances between fingers
            thumb_index_distance = cv2.norm(np.array([thumb_tip.x, thumb_tip.y]), np.array([index_tip.x, index_tip.y]))
            middle_ring_distance = cv2.norm(np.array([middle_tip.x, middle_tip.y]), np.array([ring_tip.x, ring_tip.y]))
            ring_pinky_distance = cv2.norm(np.array([ring_tip.x, ring_tip.y]), np.array([pinky_tip.x, pinky_tip.y]))

            # Calculate the average distance between fingers
            avg_distance = (thumb_index_distance + middle_ring_distance + ring_pinky_distance) / 3

            # Apply smoothing to the average distance
            avg_distance = smoothing_factor * avg_distance + (1 - smoothing_factor) * prev_avg_distance
            prev_avg_distance = avg_distance

            # Calculate new fan speed based on finger position
            new_fan_speed = int(round(np.interp(avg_distance, [0, 0.2], [fan_speed_min, fan_speed_max]) / fan_speed_interval) * fan_speed_interval)

            # Update previous fan speed with new fan speed
            prev_fan_speed = new_fan_speed
    else:
        # If no hand landmarks detected, maintain previous fan speed
        new_fan_speed = prev_fan_speed

def output(frame):

    global Device_Status
    global Device_Connected
    global device_selected
    global speed_bright
    global okh_detected
    global confirmation_received
    global new_fan_speed

    # Confirmation status
    if okh_detected:
        confirmation_received = True
    else:
        confirmation_received = False
    confirmation_text = "Confirmation: " + ("Received" if confirmation_received else "Pending")
    cv2.putText(frame, confirmation_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Device Selection
    if (device_selected == True) and (confirmation_received == True):
        Device_Connected = True
    elif (device_selected == False) and (confirmation_received == True):
        Device_Connected = False
    Device_Connection_text = "Device Connected: " + ("Connected" if Device_Connected else "Not Connected")
    cv2.putText(frame, f"{Device_Connection_text}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Gesture Recognition
    if Device_Connected == True:

        # To print the Device Status
        if confirmation_received and action_status:
            Device_Status = True
        elif confirmation_received and not action_status:
            Device_Status = False
        DeviceStatus_text = "Device Status: " + ("Turned On" if Device_Status else "Turned Off")
        # Set color based on Device Status
        color_DeviceStatus = (0, 255, 0) if Device_Status else (0, 0, 255)  # Green for turned on, Red for turned off
        # Set color for Device Status text
        cv2.putText(frame, DeviceStatus_text, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, color_DeviceStatus, 2)

        # To Display The Action Display
        status_text = "Action: " + ("ON" if action_status else "OFF")
        color_ActionStatus = (0, 255, 0) if Device_Status else (0, 0, 255)  # Green for turned on, Red for turned off
        cv2.putText(img, status_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color_ActionStatus, 2)

        # Draw fan speed bar and its accuracy
        bar_position_left = 50
        bar_position_bottom = frame.shape[0] - 50
        bar_length = int((new_fan_speed / 100) * bar_height)
        cv2.rectangle(frame, (bar_position_left, bar_position_bottom - bar_length), (bar_position_left + bar_width, bar_position_bottom), (0, 0, 255), -1)

        if (Device_Status == True) and (action_status == True):
            speed_bright = prev_fan_speed
        cv2.putText(frame, f"The Speed / Brightness Set at level: {speed_bright}%", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

while True:

    # Inside the main loop where frames are processed and displayed
    success, img = cap.read()
    if not success:
        break
    Gesture_detection(img)
    object_detection(img)

    # Process hand landmarks using MediaPipe
    ret, frame2 = cap.read()
    if not ret:
        break
    mediapipie_detection(frame2)
    Speed_Controlling(frame2)
    output(frame2)

    # Resize img to fit into frame
    resized_img = cv2.resize(img, (frame2.shape[1], frame2.shape[0]))

    # Create a combined frame to display both images
    combined_frame = cv2.addWeighted(frame2, 0.5, resized_img, 0.5, 0)

    # Display the combined frame
    cv2.imshow("Combined Functionality", combined_frame)

    # Break the loop if 'q' key is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == 27:
        break
# Release the capture
cap.release()
cv2.destroyAllWindows()