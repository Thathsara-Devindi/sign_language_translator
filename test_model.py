import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# load the trained model
model = load_model('action.h5')

# 2 words
actions = np.array(['Ayubowan', 'Ow'])

# live presiction 30 frame queue
sequence = []
sentence = []
threshold = 0.8 # more than 80% , then show in screen

cap = cv2.VideoCapture(0)

print(" AI Real-time Testing started... Press 'q' to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # collect Landmarks in live stream
    left_hand = np.zeros(21*3)
    right_hand = np.zeros(21*3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            lh = np.array([[res.x, res.y, res.z] for res in hand_lms.landmark]).flatten()
            
            if hand_label == 'Left':
                left_hand = lh
            elif hand_label == 'Right':
                right_hand = lh

            mp_drawing.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)

    # make full features array (two hands data)
    full_features = np.concatenate([left_hand, right_hand])
    
    sequence.append(full_features)
    sequence = sequence[-30:] # last 30 frames

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        predicted_action = actions[np.argmax(res)]
        confidence = res[np.argmax(res)]
        
        if confidence > threshold:
            if len(sentence) > 0:
                if predicted_action != sentence[-1]:
                    sentence.append(predicted_action)
            else:
                sentence.append(predicted_action)

        if len(sentence) > 5: 
            sentence = sentence[-5:]

        
        print(f"🔮 Predicted: {predicted_action} ({confidence*100:.2f}%)", end="\r")

    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Real-time Sign Language Translation', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()