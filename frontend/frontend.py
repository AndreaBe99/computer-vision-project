import os
from colour import Color
import math
from config import *

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, IntegerField, validators
from flask_uploads import UploadSet, IMAGES

photos = UploadSet('photos', IMAGES)
videos = UploadSet('videos', FrontendEnum.VIDEOS.value)

# :::: Form for uploading images ::::
class UploadFormImages(FlaskForm):
	photo = FileField("image_downhill",
		validators=[
			FileAllowed(photos, 'Images only!'),
			FileRequired('File was empty!')
			]
		)
	submit = SubmitField('Upload')

# :::: Form for uploading videos ::::
class UploadFormVideos(FlaskForm):
	video = FileField(
					validators=[
						FileAllowed(videos, 'Videos only!'),
						FileRequired('File was empty!')
					])
	height = IntegerField('Height',
						validators=[
							validators.InputRequired()
							])
	submit = SubmitField('Upload')

def remove_files(path):
	for file_name in os.listdir(path):
		# construct full file path
		file = path + file_name
		if os.path.isfile(file):
			# print('Deleting file:', file)
			os.remove(file)


def compute_video_suggestions(suggestions):
	move_saddle_up_down, move_saddle_forward_backward, move_handlebar_forward_backward, side = suggestions

	# Compute how much move up or down the saddle
	if move_saddle_up_down < 0:
		direction = FrontendEnum.DOWN_STR.value
		direction_icon1 = FrontendEnum.ICON_DOWN.value
	elif move_saddle_up_down > 0:
		direction = FrontendEnum.UP_STR.value
		direction_icon1 = FrontendEnum.ICON_UP.value
	str_suggestion_1 = FrontendEnum.SADDLE_STR.value + direction +\
		'%.1fcm' % abs(move_saddle_up_down)
	if math.floor(abs(move_saddle_up_down)) == 0:
		str_suggestion_1 = FrontendEnum.SADDLE_OK_HEIGHT_STR.value
		direction_icon1 = FrontendEnum.ICON_OK.value

	# Compute how much move forward or backward the saddle
	# To give the correct suggestion we need to consider the side of the bike, if left side consider east direction,
	# if right side consider west direction, so the minus sign has different meaning.
	if side == 0:
		if move_saddle_forward_backward < 0:
			direction = FrontendEnum.BACKWARD_STR.value
			direction_icon2 = FrontendEnum.ICON_RIGHT.value
		elif move_saddle_forward_backward > 0:
			direction = FrontendEnum.FORWARD_STR.value
			direction_icon2 = FrontendEnum.ICON_LEFT.value
	elif side == 1:
		if move_saddle_forward_backward < 0:
			direction = FrontendEnum.FORWARD_STR.value
			direction_icon2 = FrontendEnum.ICON_RIGHT.value
		elif move_saddle_forward_backward > 0:
			direction = FrontendEnum.BACKWARD_STR.value
			direction_icon2 = FrontendEnum.ICON_LEFT.value
	
	str_suggestion_2 = FrontendEnum.SADDLE_STR.value + direction + \
		'%.1fcm' % abs(move_saddle_forward_backward)
	if math.floor(abs(move_saddle_forward_backward)) == 0:
		direction_icon2 = FrontendEnum.ICON_OK.value
		str_suggestion_2 = FrontendEnum.SADDLE_OK_STR.value

	# Compute how much move forward or backward the handlebar
	if side == 0:
		if move_handlebar_forward_backward < 0:
			direction = FrontendEnum.BACKWARD_STR.value
			direction_icon3 = FrontendEnum.ICON_LEFT.value
		elif move_handlebar_forward_backward > 0:
			direction = FrontendEnum.FORWARD_STR.value
			direction_icon3 = FrontendEnum.ICON_RIGHT.value
	elif side == 1:
		if move_handlebar_forward_backward < 0:
			direction = FrontendEnum.FORWARD_STR.value
			direction_icon3 = FrontendEnum.ICON_RIGHT.value
		elif move_handlebar_forward_backward > 0:
			direction = FrontendEnum.BACKWARD_STR.value
			direction_icon3 = FrontendEnum.ICON_LEFT.value
	str_suggestion_3 = FrontendEnum.HANDLEBAR_STR.value + direction + \
		'%.1fcm' % abs(move_handlebar_forward_backward)
	if math.floor(abs(move_handlebar_forward_backward)) == 0:
		str_suggestion_3 = FrontendEnum.HANDLEBAR_OK_STR.value
		direction_icon3 = FrontendEnum.ICON_OK.value

	direction_icons = [direction_icon1, direction_icon2, direction_icon3]
	str_suggestions = [str_suggestion_1, str_suggestion_2, str_suggestion_3]
	return str_suggestions, direction_icons


def compute_gradient():
	red, yellow, green = Color("red"), Color("yellow"), Color("green")

	# We want a list of 100 elements, 25 red, 25 yellow, 25 green, 25 yellow
	red_to_yellow = list(red.range_to(yellow, 25))
	yellow_to_green = list(yellow.range_to(green, 25))
	green_to_yellow = list(green.range_to(yellow, 25))
	yellow_to_red = list(yellow.range_to(red, 25))
	
	colors = red_to_yellow + yellow_to_green + green_to_yellow + yellow_to_red
	return colors


def get_color(gradient, number, start, end):
	# Compute the percentage of the number in the interval [start, end]
	percentage = (number - start) / (end - start)
	# Get the color from the gradient
	return gradient[int(percentage * len(gradient))]