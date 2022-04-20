import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import os


model = torch.load('asl_detector_model.pt',map_location ='cpu')

# List of pages in the app
pages = ['Live Prediction','Text-Sign']


# Setting up the Streamlit App main settings.
st.set_page_config()
st.title('SIGN LANGUAGE RECOGNITION USING ASL&CNN')
page = st.sidebar.selectbox('Page', pages)


def play_image(char):
	path = "gestures"
	if(char == " "):
		final = os.path.join(path,  "0.jpg")
	else:
		final = os.path.join(path, char +".jpg")
		#print(final)
	gesture = Image.open(final)
	return  gesture



def app_text_to_sign(curr_index):
	name = st.text_input("Enter the text")
	if (st.button("Get gestures")):
		for char in name.title():
			img = play_image(char)
			st.image(img , width=200)
			st.write(char)

if page == 'Live Prediction':
	st.subheader('ASL Letter Check')
	st.write("""
		Press the `Run` checkbox to start your camera. Once the camera starts 
		you can start signing different ASL letters. As you sign, the prediction
		will appear in a box around your signing hand.
		""")
	run_camera = st.checkbox('Run')
	FRAME_WINDOW = st.image([])
	camera = cv2.VideoCapture(0)

	while run_camera:
		ret, frame = camera.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		mod_frame = cv2.resize(frame, (416,416), interpolation=cv2.INTER_AREA)
		results = model(mod_frame)

		# Make detections on the current frame
		FRAME_WINDOW.image(np.squeeze(results.render())) 
	
	else:
		st.write('Stopped')

elif page == 'Text-Sign':


	
    app_text_to_sign(0)

		

