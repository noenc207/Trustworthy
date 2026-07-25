

class FailureClassifier:
    def __init__(self):
        self.categories = ["hair", "bubble", "marker", "blur", "flash"]

    def classify(self, artifact_confidences: dict[str, float]) -> tuple[str, str]:
        sorted_artifacts = sorted(artifact_confidences.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_artifacts[0][0] if sorted_artifacts[0][1] > 0.1 else "none"
        secondary = sorted_artifacts[1][0] if sorted_artifacts[1][1] > 0.1 else "none"

        return primary, secondary

    def recommend_action(self, primary_cause: str) -> str:
        actions = {
            "hair": "Shave the area before retaking the image or apply digital hair removal.",
            "bubble": "Ensure proper coupling fluid application and avoid air bubbles.",
            "marker": "Clean the area of any surgical markings if possible, or use a different angle.",
            "blur": "Hold the camera steady and ensure proper focus. Retake the image.",
            "flash": "Diffuse the light source or avoid direct flash reflection. Retake without flash if lighting is sufficient.",
            "none": "No specific action required."
        }
        return actions.get(primary_cause, "Retake the image with standard protocols.")
