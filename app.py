import cv2
import pandas as pd

from vision.density_estimator import TrafficDensityEstimator
from vision.read_image_sequence import read_sequence
from nlp.incident_classifier import IncidentClassifier
from nlp.severity_mapper import SeverityMapper
from fusion.decision_engine import DecisionEngine

# ===================== CONFIG =====================
# 🔴 CHANGE THESE PATHS TO YOUR DRIVE

VISION_SEQUENCE_PATH = r"E:/dataset/vision/images/train"
NLP_DATASET_PATH     = r"E:/dataset/nlp/TWITTER DATA SET.csv"

MAX_FRAMES = 3000        # frames to process
TEXT_SAMPLE_INDEX = 0    # which row of NLP dataset to use

# ==================================================

print("========== MULTIMODAL TRAFFIC SYSTEM ==========")
print(f"[INFO] Vision data : {VISION_SEQUENCE_PATH}")
print(f"[INFO] NLP data    : {NLP_DATASET_PATH}")
print(f"[INFO] Max frames  : {MAX_FRAMES}")
print("==============================================")

# ---------- Initialize modules ----------
vision_estimator = TrafficDensityEstimator()
incident_classifier = IncidentClassifier()
severity_mapper = SeverityMapper()
decision_engine = DecisionEngine()

# ---------- Load NLP text ----------
df_text = pd.read_csv(NLP_DATASET_PATH)

sample_text = df_text.iloc[TEXT_SAMPLE_INDEX]["text"]
true_label  = df_text.iloc[TEXT_SAMPLE_INDEX]["label"]

incident = incident_classifier.classify(sample_text)
severity = severity_mapper.get_severity(incident)

print("[NLP]")
print(f"Text      : {sample_text}")
print(f"Predicted : {incident}")
print(f"Severity  : {severity}")
print("----------------------------------------------")

# ---------- Vision + Fusion ----------
frame_id = 0

for frame in read_sequence(VISION_SEQUENCE_PATH, max_frames=MAX_FRAMES):
    frame_id += 1

    density_score, density_level = vision_estimator.estimate_density(frame)

    decision = decision_engine.decide(
        density_level=density_level,
        density_score=density_score,
        incident=incident,
        severity=severity
    )

    # ---------- Overlay ----------
    cv2.putText(frame, f"Frame: {frame_id}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.putText(frame, f"Density: {density_level} ({density_score:.4f})",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    cv2.putText(frame, f"Incident: {incident}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    cv2.putText(frame, f"Decision: {decision['decision']}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)

    cv2.imshow("Multimodal Traffic Management System", frame)

    # ESC to stop
    if cv2.waitKey(30) & 0xFF == 27:
        print("[INFO] Stopped by user.")
        break

cv2.destroyAllWindows()
print("[INFO] Application finished.")
