import cv2
import numpy as np
import os
import mediapipe as mp

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Folder Setup
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['Ayubowan', 'Sthuthi', 'Ow']) # 3 words for now
no_sequences = 30 # for 1 word; 30 vedios
sequence_length = 30 # in 1 vedio; 30 frames

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)

# data collecting (Main Loop)
for action in actions:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):

            success, frame = cap.read()
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # draw landmarks 
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    # Landmarks getting to an Array (need for AI )
                    lh = np.array([[res.x, res.y, res.z] for res in hand_lms.landmark]).flatten()
            else:
                lh = np.zeros(21*3) # if no hands; then put 0

            # show details in screen 
            if frame_num == 0: 
                cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {action} Video no {sequence}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Data Collection', image)
                cv2.waitKey(2000) #2 sec
            else: 
                cv2.putText(image, f'Collecting frames for {action} Video no {sequence}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Data Collection', image)

            # Save  (.npy file)
            keypoints_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(keypoints_path, lh)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()