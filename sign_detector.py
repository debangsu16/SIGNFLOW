from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
from keras.models import model_from_json

from function import draw_styled_landmarks, extract_keypoints, mediapipe_detection


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ACTIONS = np.array(["A", "B", "C", "D", "E", "H", "R"])
ALL_ACTIONS = tuple(chr(code) for code in range(ord("A"), ord("Z") + 1))
ACTION_NAMES = tuple(ALL_ACTIONS)
DATA_PATH = BASE_DIR / "MP_Data"
PROTOTYPE_PATH = BASE_DIR / "landmark_prototypes.npz"
SEQUENCE_LENGTH = 30
WARMUP_FRAMES = 6
PREDICT_EVERY_N_FRAMES = 2
PREDICTION_WINDOW = 3
CONFIDENCE_THRESHOLD = 0.72
REQUIRED_STABLE_PREDICTIONS = 2
FAST_WARMUP_FRAMES = 3
FAST_CONFIDENCE_THRESHOLD = 0.30


def load_trained_model():
    model_path = BASE_DIR / "model.json"
    weights_path = BASE_DIR / "model.h5"
    if not model_path.exists() or not weights_path.exists():
        raise FileNotFoundError("model.json and model.h5 must be next to the app")

    model = model_from_json(model_path.read_text(encoding="utf-8"))
    model.load_weights(weights_path)
    if int(model.output_shape[-1]) != len(DEFAULT_ACTIONS):
        raise ValueError("The model output count does not match the configured labels")
    return model


def normalize_keypoints(keypoints):
    points = np.asarray(keypoints, dtype=np.float32).reshape(21, 3).copy()
    points -= points[0]
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale > 1e-6:
        points /= scale
    return points.flatten()


class LandmarkClassifier:
    def __init__(self, prototype_labels, prototypes):
        self.prototype_labels = np.asarray(prototype_labels)
        self.actions = np.unique(self.prototype_labels)
        self.prototypes = np.asarray(prototypes, dtype=np.float32)

    def predict(self, keypoints):
        normalized = normalize_keypoints(keypoints)
        prototype_distances = np.mean(
            (self.prototypes - normalized) ** 2,
            axis=1,
        )
        distances = np.asarray(
            [
                np.min(prototype_distances[self.prototype_labels == action])
                for action in self.actions
            ]
        )
        order = np.argsort(distances)
        best, second = int(order[0]), int(order[1])
        confidence = 1.0 - float(
            distances[best] / (distances[second] + 1e-9)
        )
        return str(self.actions[best]), max(0.0, min(confidence, 1.0))


def available_training_labels():
    labels = []
    if not DATA_PATH.exists():
        return labels
    for directory in DATA_PATH.iterdir():
        if directory.is_dir() and any(directory.glob("*/*.npy")):
            labels.append(directory.name.upper())
    return sorted(labels)


def dataset_sample_counts():
    return {
        label: sum(1 for _ in (DATA_PATH / label).glob("*/*.npy"))
        for label in available_training_labels()
    }


def create_capture_session(label):
    label = label.strip().upper()
    if label not in ALL_ACTIONS:
        raise ValueError("Training labels must be a single letter from A to Z")
    session_id = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")
    session_path = DATA_PATH / label / session_id
    session_path.mkdir(parents=True, exist_ok=False)
    return session_path


def save_training_sample(session_path, frame_number, keypoints):
    sample_path = Path(session_path) / f"{frame_number:04d}.npy"
    np.save(sample_path, np.asarray(keypoints, dtype=np.float32))
    return sample_path


def load_fast_classifier(force_rebuild=False):
    if PROTOTYPE_PATH.exists() and not force_rebuild:
        saved = np.load(PROTOTYPE_PATH)
        if "version" in saved and int(saved["version"][0]) == 2:
            return LandmarkClassifier(
                saved["prototype_labels"],
                saved["prototypes"],
            )

    actions = available_training_labels()
    if len(actions) < 2:
        raise ValueError("At least two trained labels are required")
    prototypes = []
    prototype_labels = []
    for action in actions:
        action_prototypes = 0
        for session_path in sorted((DATA_PATH / action).iterdir()):
            if not session_path.is_dir():
                continue
            samples = []
            for sample_path in session_path.glob("*.npy"):
                keypoints = np.load(sample_path)
                if keypoints.shape == (63,) and np.any(keypoints):
                    samples.append(normalize_keypoints(keypoints))
            if samples:
                prototypes.append(np.median(np.stack(samples), axis=0))
                prototype_labels.append(action)
                action_prototypes += 1
        if action_prototypes == 0:
            raise ValueError(f"No usable landmark data found for {action}")

    prototypes = np.stack(prototypes).astype(np.float32)
    np.savez_compressed(
        PROTOTYPE_PATH,
        version=np.asarray([2]),
        prototype_labels=np.asarray(prototype_labels),
        prototypes=prototypes,
    )
    return LandmarkClassifier(prototype_labels, prototypes)


class SignLanguageDetector:
    def __init__(self, model=None, classifier=None):
        if model is None and classifier is None:
            raise ValueError("A model or landmark classifier is required")
        self.model = model
        self.classifier = classifier
        self.warmup_frames = (
            FAST_WARMUP_FRAMES if classifier is not None else WARMUP_FRAMES
        )
        self.lock = Lock()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.reset()

    @property
    def actions(self):
        if self.classifier is not None:
            return list(self.classifier.actions)
        return list(DEFAULT_ACTIONS)

    def set_classifier(self, classifier):
        with self.lock:
            self.classifier = classifier
            self.model = None
            self.warmup_frames = FAST_WARMUP_FRAMES
            self._reset_detection_state()

    def reset(self):
        with self.lock:
            self.sentence = []
            self._reset_detection_state()

    def _reset_detection_state(self):
            self.sequence = deque(maxlen=SEQUENCE_LENGTH)
            self.predictions = deque(maxlen=PREDICTION_WINDOW)
            self.latest_label = None
            self.latest_confidence = 0.0
            self.hand_visible = False
            self.frames_with_hand = 0
            self.candidate_label = None
            self.candidate_confidence = 0.0
            self.latest_keypoints = None

    def close(self):
        self.hands.close()

    def process(self, frame, draw_status=True, classify=True):
        height, width = frame.shape[:2]
        roi_size = min(360, width, height)
        x1 = max(0, (width - roi_size) // 2)
        y1 = max(0, (height - roi_size) // 2)
        x2, y2 = x1 + roi_size, y1 + roi_size

        roi = frame[y1:y2, x1:x2]
        detected_image, results = mediapipe_detection(roi, self.hands)
        draw_styled_landmarks(detected_image, results)
        frame[y1:y2, x1:x2] = detected_image
        cv2.rectangle(frame, (x1, y1), (x2, y2), (54, 211, 153), 2)

        with self.lock:
            if results.multi_hand_landmarks:
                self.hand_visible = True
                self.frames_with_hand += 1
                self.latest_keypoints = extract_keypoints(results)
                self.sequence.append(self.latest_keypoints)
                if (
                    classify
                    and
                    self.classifier is not None
                    and len(self.sequence) >= FAST_WARMUP_FRAMES
                ):
                    self._predict_fast()
                elif (
                    classify
                    and
                    self.classifier is None
                    and len(self.sequence) >= WARMUP_FRAMES
                    and self.frames_with_hand % PREDICT_EVERY_N_FRAMES == 0
                ):
                    self._predict_sequence()
            else:
                self.hand_visible = False
                self.frames_with_hand = 0
                self.sequence.clear()
                self.predictions.clear()
                self.latest_label = None
                self.latest_confidence = 0.0
                self.candidate_label = None
                self.candidate_confidence = 0.0
                self.latest_keypoints = None

            sentence = " ".join(self.sentence[-12:]) or "Waiting for a sign"
            result = (
                f"{self.latest_label}  {self.latest_confidence:.1%}"
                if self.latest_label
                else "Show one hand inside the box"
            )

        if draw_status:
            self._draw_status(frame, result, sentence)
        return frame

    def get_state(self):
        with self.lock:
            return {
                "label": self.latest_label,
                "confidence": self.latest_confidence,
                "sentence": list(self.sentence),
                "hand_visible": self.hand_visible,
                "frames_ready": min(len(self.sequence), self.warmup_frames),
                "warmup_frames": self.warmup_frames,
                "candidate_label": self.candidate_label,
                "candidate_confidence": self.candidate_confidence,
                "keypoints": (
                    None
                    if self.latest_keypoints is None
                    else self.latest_keypoints.copy()
                ),
            }

    def _predict_fast(self):
        label, confidence = self.classifier.predict(self.sequence[-1])
        self._record_prediction(
            label,
            confidence,
            FAST_CONFIDENCE_THRESHOLD,
        )

    def _predict_sequence(self):
        observed = np.asarray(self.sequence, dtype=np.float32)
        sample_indices = np.linspace(
            0, len(observed) - 1, SEQUENCE_LENGTH
        ).round().astype(int)
        model_sequence = observed[sample_indices]
        batch = np.expand_dims(model_sequence, axis=0)
        probabilities = self.model(batch, training=False).numpy()[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        self._record_prediction(
            str(DEFAULT_ACTIONS[class_index]),
            confidence,
            CONFIDENCE_THRESHOLD,
        )

    def _record_prediction(self, label, confidence, threshold):
        self.candidate_label = label
        self.candidate_confidence = confidence
        self.predictions.append(label)

        if len(self.predictions) < REQUIRED_STABLE_PREDICTIONS:
            return

        stable_count = Counter(self.predictions).get(label, 0)
        if (
            stable_count >= REQUIRED_STABLE_PREDICTIONS
            and confidence >= threshold
        ):
            self.latest_label = label
            self.latest_confidence = confidence
            if not self.sentence or self.sentence[-1] != label:
                self.sentence.append(label)
                self.sentence = self.sentence[-30:]

    @staticmethod
    def _draw_status(frame, result, sentence):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 92), (20, 25, 35), -1)
        cv2.putText(
            frame,
            result,
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            sentence,
            (18, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (54, 211, 153),
            2,
            cv2.LINE_AA,
        )
