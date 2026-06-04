import cv2
import numpy as np
import os
import mediapipe as mp

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, # for 2 hands
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Folder Setup
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['Ayubowan', 'Ow']) # for 2 words
no_sequences = 30 # 30 vedios
sequence_length = 30 # 30 frames

# auto create the folder
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)

# Data Collecting (Main Loop)
for action in actions:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):

            success, frame = cap.read()
            if not success:
                break

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            #  Landmarks (2 hands) logic
            left_hand = np.zeros(21*3)  
            right_hand = np.zeros(21*3) 

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # MediaPipe detect left, right hands
                    hand_label = handedness.classification[0].label
                    
                    # Landmarks - flat array 
                    lh = np.array([[res.x, res.y, res.z] for res in hand_lms.landmark]).flatten()
                    
                    # puting data to hand
                    if hand_label == 'Left':
                        left_hand = lh
                    elif hand_label == 'Right':
                        right_hand = lh

                   
                    mp_drawing.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)

            full_features = np.concatenate([left_hand, right_hand])
            
            # display the data
            if frame_num == 0: 
                cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {action} Video no {sequence}', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Data Collection', image)
                cv2.waitKey(2000) # 2 min braek            else: 
                cv2.putText(image, f'Collecting frames for {action} Video no {sequence}', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Data Collection', image)

            # .npy file saving
            keypoints_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(keypoints_path, full_features)

            # 'q' 
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            continue
        break
    else:
        continue
    break

cap.release()
cv2.destroyAllWindows()