# nlp/severity_mapper.py

SEVERITY_MAP = {
    "ACCIDENT": "HIGH",
    "ROAD_CLOSURE": "HIGH",
    "CONSTRUCTION": "MEDIUM",
    "BREAKDOWN": "MEDIUM",
    "HEAVY_TRAFFIC": "LOW",
    "NORMAL": "LOW"
}

def get_severity(incident_type: str) -> str:
    return SEVERITY_MAP.get(incident_type, "LOW")
