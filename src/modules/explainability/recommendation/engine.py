

class RecommendationEngine:
    def __init__(self):
        self.default_recommendation = "Routine follow-up."

    def generate_recommendations(self, diagnosis: str, confidence: float) -> list[str]:
        recs = []
        if confidence < 0.8:
            recs.append("Seek second opinion due to model uncertainty.")

        if diagnosis.lower() == "malignant":
            recs.append("Immediate biopsy recommended.")
        else:
            recs.append("Routine follow-up in 6 months.")

        return recs
