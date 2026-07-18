import streamlit as st
import av
import numpy as np
import cv2
from keras.models import model_from_json
import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
from function import mediapipe_detection, extract_keypoints  # Your custom functions

# Load the model
with open("model.json", "r") as f:
    model_json = f.read()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define actions
actions = ['A', 'B', 'C', 'D', 'E', 'H', 'R']
threshold = 0.8

# Mediapipe hands setup
mp_hands = mp.solutions.hands

# Session states
if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "accuracy" not in st.session_state:
    st.session_state.accuracy = []
if "all_outputs" not in st.session_state:
    st.session_state.all_outputs = []

class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.predictions = []
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cropframe = img[40:400, 0:300]
        img = cv2.rectangle(img, (0, 40), (300, 400), (255, 0, 0), 2)

        image, results = mediapipe_detection(cropframe, self.hands)
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        if len(self.sequence) == 30:
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            self.predictions.append(np.argmax(res))

            if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    current_output = actions[np.argmax(res)]
                    confidence = str(round(res[np.argmax(res)] * 100, 2))

                    if len(st.session_state.sentence) == 0 or current_output != st.session_state.sentence[-1]:
                        st.session_state.sentence.append(current_output)
                        st.session_state.accuracy.append(confidence)
                        st.session_state.all_outputs.append(current_output)

                if len(st.session_state.sentence) > 1:
                    st.session_state.sentence = st.session_state.sentence[-1:]
                    st.session_state.accuracy = st.session_state.accuracy[-1:]

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Streamlit UI
st.title("📞 Real-Time Sign Language Translator in a Video Call")

# Buttons
clear = st.button("Clear Last Prediction")
if clear:
    if st.session_state.sentence:
        st.session_state.sentence.pop()
    if st.session_state.accuracy:
        st.session_state.accuracy.pop()
    if st.session_state.all_outputs:
        st.session_state.all_outputs.pop()

# WebRTC stream
webrtc_streamer(
    key="sign-language",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=SignLanguageTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

# Outputs
st.markdown(f"### Latest Output: {''.join(st.session_state.sentence)} | Accuracy: {''.join(st.session_state.accuracy)}%")
st.markdown("### Sentence Output: " + "  ".join(st.session_state.all_outputs))
