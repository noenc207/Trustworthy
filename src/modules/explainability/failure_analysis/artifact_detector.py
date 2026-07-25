
import cv2
import numpy as np


class ArtifactDetector:
    def __init__(self):
        self.thresholds = {"hair": 0.5, "bubble": 0.5, "marker": 0.5, "blur": 0.5, "flash": 0.5}

    def detect_hair(self, image: np.ndarray) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hair_pixels = sum(cv2.contourArea(c) for c in contours)
        total_pixels = image.shape[0] * image.shape[1]
        return min(1.0, float(hair_pixels) / float(total_pixels) * 50.0)

    def detect_bubble(self, image: np.ndarray) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50)
        if circles is not None:
            return min(1.0, len(circles[0]) / 20.0)
        return 0.0

    def detect_marker(self, image: np.ndarray) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            return min(1.0, len(lines) / 30.0)
        return 0.0

    def detect_blur(self, image: np.ndarray) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        confidence = max(0.0, min(1.0, 1.0 - (variance / 500.0)))
        return float(confidence)

    def detect_flash(self, image: np.ndarray) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        num_white = np.sum(gray > 240)
        total = gray.shape[0] * gray.shape[1]
        return min(1.0, float(num_white) / float(total) * 10.0)

    def detect_all(self, image: np.ndarray) -> dict[str, float]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {"hair": 0.0, "bubble": 0.0, "marker": 0.0, "blur": 0.0, "flash": 0.0}

        return {
            "hair": self.detect_hair(image),
            "bubble": self.detect_bubble(image),
            "marker": self.detect_marker(image),
            "blur": self.detect_blur(image),
            "flash": self.detect_flash(image)
        }
