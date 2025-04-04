import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from math import hypot

# Initialize MediaPipe Hand and Face Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get system audio interface
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol_range = volume.GetVolumeRange()
    min_vol = vol_range[0]
    max_vol = vol_range[1]
except Exception as e:
    print(f"Error accessing audio system: {e}")
    exit(1)

# Start Webcam Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit(1)

double_blink_counter = 0
blink_count = 0
blink_time_threshold = 20  # Number of frames within which blinks should occur
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
            distance = hypot(x2 - x1, y2 - y1)
            vol = np.interp(distance, [30, 250], [min_vol, max_vol])
            vol = min(max(vol, min_vol), max_vol)
            volume.SetMasterVolumeLevel(vol, None)
            vol_percentage = int(np.interp(vol, [min_vol, max_vol], [0, 100]))
            
            # Draw volume text and line
            cv2.putText(frame, f"Volume: {vol_percentage}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw graph representation of volume
            bar_x, bar_y = 50, 100
            bar_height = 200
            filled_height = int((vol_percentage / 100) * bar_height)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 30, bar_y + bar_height), (0, 255, 0), 2)
            cv2.rectangle(frame, (bar_x, bar_y + bar_height - filled_height), (bar_x + 30, bar_y + bar_height), (0, 255, 0), -1)
    
    # Detect eye blinks for changing media
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            left_eye_top = face_landmarks.landmark[159].y
            left_eye_bottom = face_landmarks.landmark[145].y
            right_eye_top = face_landmarks.landmark[386].y
            right_eye_bottom = face_landmarks.landmark[374].y
            
            left_eye_ratio = abs(left_eye_top - left_eye_bottom)
            right_eye_ratio = abs(right_eye_top - right_eye_bottom)
            
            if left_eye_ratio < 0.02 and right_eye_ratio < 0.02:
                if frame_counter - blink_count <= blink_time_threshold:
                    double_blink_counter += 1
                else:
                    double_blink_counter = 1
                blink_count = frame_counter
            
            if double_blink_counter >= 2:
                pyautogui.press('nexttrack')  # Skip to the next song or video
                double_blink_counter = 0
    
    frame_counter += 1
    cv2.imshow("Hand Gesture Volume & Eye Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
