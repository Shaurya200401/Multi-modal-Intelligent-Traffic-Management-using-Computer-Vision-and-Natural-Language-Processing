import sys
import os

# Add project root to sys.path to allow running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# nlp/test.py
from nlp.preprocess import load_clean_and_label
from nlp.incident_classifier import IncidentClassifier
from nlp.severity_mapper import get_severity

CSV_PATH = r"D:/study/major proj/project files/data/TWITTER DATA SET.csv"

print("[INFO] Loading and cleaning data...")
df = load_clean_and_label(CSV_PATH)



print("[INFO] Training classifier...")
classifier = IncidentClassifier()
classifier.train(df["clean_text"], df["label"])
classifier.save()

print("[INFO] Model trained and saved.")

# ---- Test on sample text ----
sample_texts = [
    "accident reported on highway near flyover",
    "road resurfacing works expect delays",
    "stationary vehicle causing traffic"
]

classifier.load()

print("\n[INFO] Testing NLP module:")
for text in sample_texts:
    incident = classifier.predict(text)
    severity = get_severity(incident)
    print(f"Text      : {text}")
    print(f"Incident  : {incident}")
    print(f"Severity  : {severity}")
    print("-" * 40)
