import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Volume Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Mediapipe Hand Model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class HandVolumeController(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame):
        # Read frame from webcam
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Get thumb and index finger coordinates
                h, w, _ = img.shape
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                # Draw circle and line between thumb and index
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Calculate the distance between thumb and index
                length = math.hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [50, 200], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                # Add volume level text
                cv2.putText(
                    img,
                    f"Volume: {int(np.interp(length, [50, 200], [0, 100]))}%",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        return img


# Streamlit UI
st.title("Hand Gesture Volume Control")
st.write("Control your system volume using your hand gestures.")

webrtc_streamer(
    key="hand-volume-control",
    video_transformer_factory=HandVolumeController,
    media_stream_constraints={"video": True, "audio": False},
)
