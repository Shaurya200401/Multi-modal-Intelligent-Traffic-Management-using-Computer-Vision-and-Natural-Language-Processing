class TrafficAdvisoryGenerator:
    def __init__(self):
        pass

    def generate(
        self,
        area_name: str,
        traffic_state: str,
        incident_type: str = None,
        severity: str = "LOW",
        alternate_route: str = None
    ) -> str:

        # Determine problem description
        if incident_type and incident_type != "NORMAL":
            problem = incident_type.replace("_", " ").lower()
        else:
            problem = traffic_state.replace("_", " ").lower()

        # Severity phrasing
        if severity == "HIGH":
            severity_phrase = "severe"
        elif severity == "MEDIUM":
            severity_phrase = "moderate"
        else:
            severity_phrase = "minor"

        # Route advice
        if alternate_route:
            route_advice = f", advised to take {alternate_route} route"
        else:
            route_advice = ", expect delays"

        advisory = (
            f"{area_name} has {severity_phrase} {problem}"
            f"{route_advice}."
        )

        return advisory
