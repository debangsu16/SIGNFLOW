import traceback

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from sign_detector import SignLanguageDetector, load_trained_model


@st.cache_resource
def load_model():
    return load_trained_model()


class SignLanguageVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = SignLanguageDetector(load_model())
        self.last_error = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        try:
            processed = self.detector.process(image)
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            traceback.print_exc()
            processed = image
            cv2.rectangle(
                processed,
                (0, 0),
                (processed.shape[1], 70),
                (20, 25, 35),
                -1,
            )
            cv2.putText(
                processed,
                f"Detection error: {str(exc)[:70]}",
                (18, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 80, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(processed, format="bgr24")


st.set_page_config(page_title="Sign Language Detector", layout="wide")
st.title("Real-Time Sign Language Detector")
st.caption(
    "Allow camera access, keep one hand inside the green box, and hold each sign "
    "steady for about one second."
)

try:
    load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

context = webrtc_streamer(
    key="sign-language-detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageVideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 24, "max": 30},
        },
        "audio": False,
    },
    async_processing=False,
    video_html_attrs={
        "autoPlay": True,
        "controls": False,
        "muted": True,
    },
)

controls, information = st.columns([1, 3])
with controls:
    if st.button("Clear sentence", use_container_width=True):
        if context.video_processor:
            context.video_processor.detector.reset()
        st.rerun()
with information:
    st.info("This model currently recognizes: A, B, C, D, E, H, and R.")

if context.video_processor and context.video_processor.last_error:
    st.error(f"Video processing error: {context.video_processor.last_error}")

st.markdown(
    """
**Tips**

- Use good front lighting and a plain background.
- Keep only one hand visible.
- The sentence adds a letter only after the prediction is stable.
- Move your hand away briefly before repeating the same letter.
"""
)
