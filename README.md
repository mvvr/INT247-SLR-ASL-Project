# Sign Language Recognition using ASL and CNN

## Context:
American Sign Language (ASL) is the most popular standard for sign language in North America. However, it can be tough for those who do not know sign language to communicate with those who cannot communicate easily without it. This project serves as an introduction into the world of using ML to recognize sign language.

## Objective:
The goal of this project was to create a convolutional neural network to recognize the  ASL alphabet.
additionally build an web app with Streamlit

## Dataset:
https://www.kaggle.com/grassknoted/asl-alphabet


The dataset contains 87,000 200x200 pixel images; 3000 images for each letter of the alphabet, in addition to space, delete, and nothing. 

## Relevant Packages:
* Pytorch: Used to create and train CNN
* OpenCV: Used for image processing
* Matplotlib and Seaborn for image visualization
* Streamlit for Web app

## Results:
The model achieved an accuracy of 97% after completing 17 epochs.

Confusion Matrix:
![alt text]()

Test Data Results:
![alt text]()

