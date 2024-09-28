# main.py

import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import face_recognition
import cv2
import numpy as np
import uvicorn
import os
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition to integrate with Flutter app.",
    version="1.0.0"
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Face Recognition API. Use /docs for API documentation."}

# Load the known face image and encode it at startup
KNOWN_FACES_DIR = "known_faces"
KNOWN_FACE_FILENAME = "mahesh.jpeg"

def load_known_face():
    known_face_path = os.path.join(KNOWN_FACES_DIR, KNOWN_FACE_FILENAME)
    if not os.path.exists(known_face_path):
        raise FileNotFoundError(f"Known face image not found at {known_face_path}")
    
    known_image = face_recognition.load_image_file(known_face_path)
    encodings = face_recognition.face_encodings(known_image)
    if not encodings:
        raise ValueError("No faces found in the known face image.")
    return encodings[0]

try:
    known_face_encoding = load_known_face()
    logger.info("Known face encoding loaded successfully.")
except Exception as e:
    logger.error(f"Error loading known face: {e}")
    known_face_encoding = None

@app.post("/recognize-face/")
async def recognize_face(file: UploadFile = File(...)):
    logger.info("Received a request for face recognition.")
    
    if known_face_encoding is None:
        logger.error("Known face encoding not loaded.")
        raise HTTPException(status_code=500, detail="Known face encoding not loaded.")

    # Validate file type
    if not file.content_type.startswith('image/'):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    try:
        # Read the uploaded image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            logger.warning("Could not decode the image.")
            raise HTTPException(status_code=400, detail="Could not decode the image.")

        # Resize image for faster processing
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        logger.info("Image resized successfully.")

        # Convert BGR (OpenCV format) to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        logger.info("Image color converted from BGR to RGB.")

        # Detect faces and compute encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        logger.info(f"Detected {len(face_locations)} face(s) in the image.")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        logger.info(f"Computed encodings for {len(face_encodings)} face(s).")

        if not face_encodings:
            logger.info("No faces detected in the uploaded image.")
            return JSONResponse(content={"matches": [], "message": "No faces detected."})

        results = []
        for face_location, encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces([known_face_encoding], encoding)
            face_distances = face_recognition.face_distance([known_face_encoding], encoding)
            best_match_index = np.argmin(face_distances)
            match = matches[best_match_index]
            distance = float(face_distances[best_match_index])
            results.append({
                "match": bool(match),       # Convert numpy.bool_ to native bool
                "distance": distance
            })
            logger.info(f"Match: {match}, Distance: {distance}")

            # Draw bounding box and label on the original image
            top, right, bottom, left = face_location
            # Scale back up face locations since the image was resized to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw rectangle around face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0) if match else (0, 0, 255), 2)

            # Prepare label
            label = "Match" if match else "No Match"

            # Calculate label position
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            label_top = max(top, label_size[1] + 10)
            cv2.rectangle(image, (left, label_top - label_size[1] - 10), 
                          (left + label_size[0] + 10, label_top + 5), 
                          (0, 255, 0) if match else (0, 0, 255), cv2.FILLED)
            cv2.putText(image, label, (left + 5, label_top - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Encode the image to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # Use jsonable_encoder to ensure all data types are JSON-serializable
        json_compatible_results = jsonable_encoder({
            "matches": results,
            "message": "Face recognition completed.",
            "image": jpg_as_text  # Include the processed image as Base64 string
        })

        return JSONResponse(content=json_compatible_results)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during face recognition: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
