import cv2
from fastapi import FastAPI

app = FastAPI()

# Load model at startup
model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_face_cords(img, cords):
    if cords is None:
        return None

    x, y, w, h = cords
    padding = int(max(w, h) * 0.9)  # 90% padding

    x1 = max(x - padding, 0)
    x2 = min(x + w + padding, img.shape[1])
    y1 = max(y - padding, 0)
    y2 = min(y + h + padding, img.shape[0])

    return x1, y1, x2 - x1, y2 - y1  # return cropped coords as (x, y, w, h)

def detect_face(model, img):
    gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray_scaled, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("No faces detected.")
        return None
    else:
        print(f"{len(faces)} face(s) detected.")
        return faces[0]  # Returns the first face found

@app.get("/api/cords")
def get_cords():
    img = cv2.imread("human.jpg")
    if img is None:
        return {"error": "Image not found"}

    cords = detect_face(model, img)
    final_cords = get_face_cords(img, cords)

    if final_cords is None:
        return {"error": "No face detected"}

    x, y, w, h = final_cords
    return {
    "x": int(x),
    "y": int(y),
    "width": int(w),
    "height": int(h)
}

