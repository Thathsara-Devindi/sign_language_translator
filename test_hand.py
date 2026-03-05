import cv2
import mediapipe as mp

# MediaPipe solutions  access 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("Camera opening ... To stop press 'q'")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("no data coming from camera")
        break

    # Mirror effect & RGB 
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # AI processing
    results = hands.process(image_rgb)

    # to draw hand landmarks 
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking Test', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()