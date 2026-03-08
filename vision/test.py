import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from vision.density_estimator import TrafficDensityEstimator
from vision.image_sequence import read_sequence
from vision.traffic_temporal_model import TrafficTemporalModel
from analysis.traffic_logger import TrafficLogger
from analysis.plot_traffic_states import plot_traffic_states
from decision.congestion_warning import CongestionWarning
from decision.advisory_generator import TrafficAdvisoryGenerator
from nlp.incident_classifier import IncidentClassifier
from nlp.severity_mapper import get_severity
from nlp.live_text_feed import TrafficTextFeed


# ===================== CONFIG =====================
SEQUENCE_PATH = r"E:\dataset\vision\images\train"
MAX_FRAMES = 3000
DISPLAY_SCALE = 0.7
# ==================================================

AREA_NAME = "[Place]"
ALTERNATE_ROUTE = "[Alternate Route]"
TEXT_CSV_PATH = r"D:\study\major proj\project files\data\TWITTER DATA SET.csv"


def resize_for_display(frame, scale=0.7):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def draw_multiline_text(img, text, start_y, line_height=25):
    x = 20
    y = start_y
    for line in text.split(", "):
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )
        y += line_height


# -------- INITIALIZATION --------
estimator = TrafficDensityEstimator()
temporal_model = TrafficTemporalModel(window_size=15)
logger = TrafficLogger()
warning_system = CongestionWarning()
advisory_generator = TrafficAdvisoryGenerator()

incident_classifier = IncidentClassifier()
incident_classifier.load()

text_feed = TrafficTextFeed(TEXT_CSV_PATH)

previous_traffic_state = None
current_text = ""
current_incident = "NORMAL"
current_severity = "LOW"

print("[INFO] Starting vision + NLP test...")
frame_id = 0


# -------- MAIN LOOP --------
try:
    for frame in read_sequence(SEQUENCE_PATH, max_frames=MAX_FRAMES):
        frame_id += 1

        density_score, density_level, fg_mask, edge_mask = estimator.estimate_density(frame)
        traffic_state = temporal_model.classify_state(density_score)
        logger.log(frame_id, density_score, traffic_state)

        # ---------- EVENT-DRIVEN NLP ----------
        if traffic_state != previous_traffic_state:
            current_text = text_feed.get_next()
            current_incident = incident_classifier.predict(current_text)
            current_severity = get_severity(current_incident)

            print("\n[EVENT] Traffic state changed")
            print(f"[NLP] Text     : {current_text}")
            print(f"[NLP] Incident : {current_incident}")
            print(f"[NLP] Severity : {current_severity}")

            previous_traffic_state = traffic_state

        # ---------- ADVISORY ----------
        advisory_text = advisory_generator.generate(
            area_name=AREA_NAME,
            traffic_state=traffic_state,
            incident_type=current_incident,
            severity=current_severity,
            alternate_route=ALTERNATE_ROUTE
        )

        warning = warning_system.update(traffic_state)

        # ---------- DISPLAY ----------
        display_frame = frame.copy()

        cv2.putText(display_frame, f"Frame: {frame_id}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(display_frame,
                    f"Density: {density_level} ({density_score:.3f})",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(display_frame,
                    f"Traffic State: {traffic_state}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if warning:
            cv2.putText(display_frame, warning, (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        draw_multiline_text(display_frame, advisory_text, start_y=160)

        if current_text:
            cv2.putText(display_frame,
                        f"NLP Input: {current_text[:60]}...",
                        (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        cv2.imshow("Traffic Frame (Final Output)",
                   resize_for_display(display_frame, DISPLAY_SCALE))

        cv2.imshow("Foreground Mask (Moving Vehicles)",
                   resize_for_display(fg_mask, DISPLAY_SCALE))

        cv2.imshow("Optical Flow (Motion Direction)",
                   resize_for_display(estimator.flow.draw_flow(frame), DISPLAY_SCALE))

        cv2.imshow("Static Occupancy (Edges)",
                   resize_for_display(edge_mask, DISPLAY_SCALE))

        if cv2.waitKey(30) & 0xFF == 27:
            print("[INFO] Stopped by user.")
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    cv2.destroyAllWindows()
    plot_traffic_states(logger)
    print("[INFO] Vision + NLP test finished.")
