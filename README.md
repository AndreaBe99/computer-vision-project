# computer-vision-project
Project for Computer Vision Course from La Sapienza Univesrity of Rome.

This project aims to carry out two main tasks.

- The first task concerns the issue of Bike Fitting, i.e. the initial bike configuration procedure, which is very important for biomechanical reasons, since an incorrect configuration can lead to problems with the cyclist's joints in the long term.
However, this procedure is very difficult to perform alone, so our goal is to use a short video, in which to trace the cyclist's posture while pedaling on his bike, in order to calculate the distances and angles between the various joints to understand if the configuration is correct or make suggestions, such as raising or lowering the saddle, or moving it forward or backward.

- The second Task has the objective of correcting the posture of MTB cyclists during descents on rough Trails, in fact given an input side image of the cyclist traveling downhill, the program identifies the wheels of the bicycle and the Key Points of the cyclist to calculate if the center of gravity of the latter is located inside the two wheels. This is because if the center of gravity is too far back, grip on the front will be lost, while if it is too far forward, the cyclist could fall forward if he encounters an obstacle such as a stone or a root.


## Folder Structure

```
.
├── backend                             # Computer Vision Tasks
│   ├── bike_fitting        
│   │   └── test.py                     # Bike Fitting Task
│   └── mtb_down_hill       
│       ├── yolo_object_detection
│       │   ├── yolov3.cgf              # Config File for YOLO
│       │   ├── yolov3.txt              # Classes List for YOLO
│       │   └── yolov3.weights          # Weights for YOLO (download it!)
│       ├── bike_detector.py            # MTB Downhill Task
│       └── yolo_object_detection.py    # Bike Detection Task
├── frontend                            
│   └── frontend.py                     # Process outputs of backend
├── doc                                 # Documentation files
│   └── etc.
├── static                              # Web Site static resources
│   └── etc.
├── templates                           # Web Site HMTL Code
│   ├── bike_fitting.html
│   ├── downhill.html
│   └── index.html
├── app.py                              # Flask root file
├── config.py                           # Configuration file for Flask and Enum
├── Dockerfile
├── requirements.txt
└── README.md
```

## Step to deploy on Google Cloud Platform

1. Create a bucket if it not exists.

    a. Creating a GCS Bucket for Your Files

        gsutil mb gs://cv-project-bucket

    b. Adding Permissions

        gsutil defacl set public-read gs://cv-project-bucket

2. Download repository
    ```
        git clone https://github.com/AndreaBe99/computer-vision-project.git 
    ```

3. Download the .weights file.
    ```
    cd computer-vision-project/backend/mtb_downhill/yolo_object_detection/
    wget https://pjreddie.com/media/files/yolov3.weights
    ```

4. Download and then rename Google Account Credential  (It is private):

    ```
    mv source_file GAC.json
    ```

5. Export the Project ID
    ```
    export GOOGLE_CLOUD_PROJECT=computer-vision-project-372315
    ```

6. Set the project
    ```
    gcloud config set project computer-vision-project-372315
    ```

7. Build Docker Image, (i is the version number of the image)
    ```
    docker build --tag cv-website:v[i] .
    ```

8. Run locally
    ```
    docker run --rm -p 8080:8080 cv-website:v[i]
    ```

9. Deploy on Google Cloud Platform

    ```
    gcloud builds submit --tag gcr.io/${GOOGLE_CLOUD_PROJECT}/computer-vision-website

    gcloud run deploy cv-website --image gcr.io/${GOOGLE_CLOUD_PROJECT}/computer-vision-website
    ```

## Step to reproduce it loacally

1. Download repository:
    ```
        git clone https://github.com/AndreaBe99/computer-vision-project.git 
    ```

2. Download the .weights file:
    ```
    cd computer-vision-project/backend/mtb_downhill/yolo_object_detection/
    wget https://pjreddie.com/media/files/yolov3.weights
    ```

3. Download and then rename Google Account Credential  (It is private):
    ```
    mv source_file GAC.json
    ```

4. Create venv and activate it:
    ```
    python3 -m venv cv-env

    source cv-env/bin/activate
    ```

5. Install requirements:
    ```
    pip3 install -r requirements.txt
    ```

6. Install ffmpeg:
    ```
    apt-get -y update
    apt-get -y upgrade
    apt-get install -y ffmpeg
    ```

7. Run:
    ```
    python3 app.py
    ```