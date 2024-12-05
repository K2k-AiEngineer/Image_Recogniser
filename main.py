from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from io import BytesIO

app = FastAPI()

# Load models once at startup for efficiency
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
embedder = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")


def get_face_embeddings(image_data: bytes, model, net):
    # Decode image from bytes
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    h, w = image.shape[:2]

    # Prepare image for the face detector
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Iterate through detections and extract face regions
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]

            # Generate embeddings for the detected face
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            model.setInput(face_blob)
            embeddings = model.forward()
            return embeddings.flatten()

    return None


@app.post("/compare_faces/")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # Read image bytes
        image1_data = await file1.read()
        image2_data = await file2.read()

        # Generate embeddings for both images
        embeddings1 = get_face_embeddings(image1_data, embedder, face_net)
        embeddings2 = get_face_embeddings(image2_data, embedder, face_net)

        if embeddings1 is None:
            raise HTTPException(status_code=400, detail="No face detected in the first image.")
        if embeddings2 is None:
            raise HTTPException(status_code=400, detail="No face detected in the second image.")

        # Compare embeddings using cosine similarity
        similarity = 1 - cosine(embeddings1, embeddings2)
        threshold = 0.5  # Similarity threshold

        if similarity > threshold:
            return JSONResponse(content={"similarity": similarity, "result": "The faces are of the same person."})
        else:
            return JSONResponse(content={"similarity": similarity, "result": "The faces are of different people."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run using `uvicorn filename:app --reload`
