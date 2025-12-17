const int ledPin = 7;
const int ledPin2 = 6; // Assuming the LED is connected to pin 6

void setup() {
  pinMode(ledPin, OUTPUT); // Set pin 7 as an output
  pinMode(ledPin2, OUTPUT); // Set pin 6 as an output
  Serial.begin(9600); // Begin serial communication
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();
    if (receivedChar == 'H') {
      digitalWrite(ledPin, HIGH); // Turn on the first LED
      Serial.println("LED 1 turned ON");
    } else if (receivedChar == 'L') {
      digitalWrite(ledPin, LOW); // Turn off the first LED
      Serial.println("LED 1 turned OFF");
    } else if (receivedChar == 'H2') {
      digitalWrite(ledPin2, HIGH); // Turn on the second LED
      Serial.println("LED 2 turned ON");
    } else if (receivedChar == 'L2') {
      digitalWrite(ledPin2, LOW); // Turn off the second LED
      Serial.println("LED 2 turned OFF");
    }
  }
}
