# Image_Recogniser/Face Comparison API

This is a FastAPI-based application that compares two uploaded face images and determines if the faces belong to the same person. It uses OpenCV's DNN module for face detection and embedding generation.

## Features

- Accepts two image files as input.
- Detects faces in the images using a pre-trained Caffe model.
- Generates face embeddings using a pre-trained OpenFace Torch model.
- Compares embeddings using cosine similarity.
- Returns a JSON response indicating whether the faces match.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/face-comparison-api.git
   cd face-comparison-api
Create a virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the dependencies:

pip install -r requirements.txt
Download the required pre-trained models:

Caffe Face Detection Model
Caffe Weights
OpenFace Torch Model
Place these files in the /content/ directory.

Usage
Start the server:

bash
Copy code
uvicorn main:app --reload
Access the API documentation at:

arduino
Copy code
http://127.0.0.1:8000/docs
Use the /compare_faces/ endpoint to upload two images. The API will return a JSON response with similarity scores and results.

Example Request
Using curl:

bash
Copy code
curl -X POST "http://127.0.0.1:8000/compare_faces/" \
-F "file1=@path/to/first/image.jpg" \
-F "file2=@path/to/second/image.jpg"
Response
The API returns a JSON object:

json
Copy code
{
    "similarity": 0.75,
    "result": "The faces are of the same person."
}
Requirements
Python 3.7+
Pre-trained Caffe and Torch models
