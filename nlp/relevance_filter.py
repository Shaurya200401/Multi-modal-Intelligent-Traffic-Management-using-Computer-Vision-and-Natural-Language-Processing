TRAFFIC_KEYWORDS = [
    "traffic", "accident", "road", "lane", "closed", "closure",
    "congestion", "jam", "delay", "vehicle", "stationary",
    "breakdown", "construction", "works"
]

def is_traffic_relevant(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in TRAFFIC_KEYWORDS)
