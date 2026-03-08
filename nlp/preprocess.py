import re
import pandas as pd


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def assign_incident_label(text: str) -> str:
    if any(word in text for word in ["accident", "crash", "collision"]):
        return "ACCIDENT"
    if any(word in text for word in ["road closed", "blocked", "closure"]):
        return "ROAD_CLOSURE"
    if any(word in text for word in ["work", "resurfacing", "construction"]):
        return "CONSTRUCTION"
    if any(word in text for word in ["stationary", "breakdown", "broken"]):
        return "BREAKDOWN"
    if any(word in text for word in ["heavy traffic", "congestion", "traffic jam"]):
        return "HEAVY_TRAFFIC"
    return "NORMAL"


def load_clean_and_label(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = df[["TWEET"]].dropna()
    df["clean_text"] = df["TWEET"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 5]

    df["label"] = df["clean_text"].apply(assign_incident_label)
    return df
