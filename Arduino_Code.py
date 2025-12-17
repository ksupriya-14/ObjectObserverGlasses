import serial
import time
import cv2
from Detection_with_Arduino import Result

# Create video capture object
cap = cv2.VideoCapture(0)
cap.set(3, 1980)
cap.set(4, 1080)

# Define the serial port (change it according to your system)
ser = serial.Serial('COM8', 9600)  # Change 'COM6' to the appropriate port on your system

# Wait for the serial connection to be2 sec established
time.sleep(2)

# Main loop         
while True:
    # Receive confirmation from sample code
    confirmation_received = Result(cap)

    if confirmation_received:
        ser.write(b'H')  # Turn on the LED
        print("LED ON")
    elif not confirmation_received:
        ser.write(b'L')  # Turn off the LED
        print("LED OFF")





