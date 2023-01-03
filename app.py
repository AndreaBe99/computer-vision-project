import cv2
import numpy as np
from urllib.request import urlopen

from tempfile import TemporaryFile
import io
from PIL import Image

from datetime import timedelta
from flask import Flask, request, render_template, send_from_directory, url_for
from flask_uploads import configure_uploads
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# My libraries
from config import *
from frontend.frontend import *
from backend.mtb_downhill.bike_detector import pipeline as bike_detector_pipeline
from backend.bike_fitting.capture_video_test import pipeline as bike_fitting_pipeline

from google.cloud import storage


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS


app = Flask(__name__)
app.config["SECRET_KEY"] = FrontendEnum.SECRET_KEY.value
app.config['UPLOADED_PHOTOS_DEST'] = FrontendEnum.UPLOADED_PHOTOS_DEST.value
app.config['UPLOADED_VIDEOS_DEST'] = FrontendEnum.UPLOADED_VIDEOS_DEST.value
app.config['MAX_CONTENT_LENGTH'] = FrontendEnum.MAX_CONTENT_LENGTH.value

# Set the upload folders
configure_uploads(app, photos)
configure_uploads(app, videos)

def upload_file_to_cloud_storage(file, filename):
	"""Upload a file to the bucket."""
	filename = secure_filename(filename)
	gcs = storage.Client()
	bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
	blob = bucket.blob(filename)
	blob.upload_from_string(
		file.read(),
		content_type=file.content_type
	)
	blob.make_public()
	return blob.media_link

@app.route('/', )
@app.route('/index.html', )
def upload_form():
	return render_template('index.html')

@app.route('/downhill.html', methods=['GET', 'POST'])
def upload_image():
	form = UploadFormImages()
	if form.validate_on_submit():
		remove_files(app.config['UPLOADED_PHOTOS_DEST'])
		# upload to cloud storage
		gcloud_url = upload_file_to_cloud_storage(form.photo.data, form.photo.data.filename)
		
		# Process the image
		resp = urlopen(gcloud_url)
		img = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(img, cv2.IMREAD_COLOR)

		result, suggestion, color = bike_detector_pipeline(image)

		if result is None:
			gcloud_url = None
			suggestion = None
			color = None
			return render_template('downhill.html', 
									form=form, 
									file_url=gcloud_url, 
									suggestion=suggestion,
									color=color)
		
		# Convert the result to a file, and the latter to a FileStorage 
		# object to be uploaded to cloud storage
		img = Image.fromarray(result)
		result_file = io.BytesIO()
		img.save(result_file, format="jpeg")
		result_file.seek(0)
		result_file = FileStorage(result_file)
		
		# result_path = app.config['UPLOADED_PHOTOS_DEST']+'result.png'
		# cv2.imwrite(result_path, result)

		gcloud_url_result = upload_file_to_cloud_storage(result_file, filename="result.jpg")
	else:
		gcloud_url_result = None
		suggestion = None
		color = None
	return render_template('downhill.html',
                        form=form,
                        file_url=gcloud_url_result,
                        suggestion=suggestion,
                        color=color)


@app.route('/bike_fitting.html', methods=['GET', 'POST'])
def upload_video():
	form = UploadFormVideos()
	if form.validate_on_submit():
		# Remove all files in the folder
		remove_files(app.config['UPLOADED_VIDEOS_DEST'])

		# upload to cloud storage
		gcloud_url = upload_file_to_cloud_storage(form.video.data, form.video.data.filename)
		
		# Process the video
		video = cv2.VideoCapture(gcloud_url)

		suggestions, angles = bike_fitting_pipeline(video, int(form.height.data))
		
		# Elaborate the results
		suggestions, direction_icons = compute_video_suggestions(suggestions)
		
		# Unpack the results
		suggestion1, suggestion2, suggestion3 = suggestions
		direction_icon1, direction_icon2, direction_icon3 = direction_icons
		knee_angle, ankle_angle, torso_angle = angles
		
		# Cast the angles to int
		knee_angle = int(knee_angle)
		ankle_angle = int(ankle_angle)
		torso_angle = int(torso_angle)

		# Compute the color gradient for the angle to visualize it on the html
		gradient = compute_gradient()

		# We subtract 5 to the angles to have a better position for the div in the html
		margin_knee = knee_angle - AnglesRange.KNEE.value[0] - 5
		margin_ankle = ankle_angle - AnglesRange.ANKLE.value[0] - 5
		margin_torso = torso_angle - AnglesRange.TORSO.value[0] - 5

		color_knee = get_color(gradient, 
								knee_angle,
								AnglesRange.KNEE.value[0], 
								AnglesRange.KNEE.value[2])
		color_ankle = get_color(gradient, 
								ankle_angle,
								AnglesRange.ANKLE.value[0], 
								AnglesRange.ANKLE.value[2])
		color_torso = get_color(gradient, 
								torso_angle,
								AnglesRange.TORSO.value[0], 
								AnglesRange.TORSO.value[2])

		video_url = BikeFittingEnum.VIDEO_LANDMARKS_PATH_MP4.value
		photo_url = BikeFittingEnum.PHOTO_PEDALING_PROGRESS_PATH.value

	else:
		video_url, photo_url = None, None
		suggestion1, suggestion2, suggestion3 = None, None, None
		direction_icon1, direction_icon2, direction_icon3 = None, None, None
		knee_angle, ankle_angle, torso_angle = None, None, None
		margin_knee, margin_ankle, margin_torso = None, None, None
		color_knee, color_ankle, color_torso = None, None, None

	return render_template('bike_fitting.html',
						   form=form,
						   video_url=video_url, photo_url=photo_url,
						   suggestion1=suggestion1, suggestion2=suggestion2, suggestion3=suggestion3,
						   direction_icon1=direction_icon1, direction_icon2=direction_icon2, direction_icon3=direction_icon3,
						   knee_angle=knee_angle, margin_knee=margin_knee, color_knee=color_knee,
						   ankle_angle=ankle_angle, margin_ankle=margin_ankle, color_ankle=color_ankle,
						   torso_angle=torso_angle, margin_torso=margin_torso, color_torso=color_torso)


if __name__ == "__main__":
	app.run(debug=True)