🔢 Handwritten Digit Recognizer

A machine learning web application that recognizes handwritten digits from 0 to 9.

📌 Project Overview

The Handwritten Digit Recognizer is a machine learning application developed using Python and Scikit-learn.

The application takes handwritten digit images from the Scikit-learn Digits Dataset and predicts which number the image represents.

The application is deployed as an interactive web application using Streamlit.

🎯 Project Objective

The main objective of this project is to understand how machine learning can be used for image classification.

The project demonstrates:

Dataset loading

Data preprocessing

Training and testing data splitting

Machine learning model training

Model prediction

Model accuracy evaluation

Interactive web application development

Cloud deployment

🧠 Machine Learning Algorithm

The project uses the K-Nearest Neighbors (KNN) classification algorithm.

KNN predicts the class of a new data point by comparing it with nearby examples from the training dataset.

For this project, KNN is used to classify handwritten digits from 0 to 9.

📊 Dataset

The project uses the built-in Digits Dataset provided by Scikit-learn.

The dataset contains images of handwritten digits from 0 to 9.

Each image is represented numerically so that a machine learning algorithm can process it.

⚙️ Technologies Used

Python

Scikit-learn

NumPy

Matplotlib

Streamlit

GitHub

🔄 Project Workflow
Handwritten Digit Dataset
          ↓
Data Preparation
          ↓
Train/Test Split
          ↓
KNN Model Training
          ↓
Model Evaluation
          ↓
Digit Prediction
          ↓
Streamlit Web Application
          ↓
Cloud Deployment

📈 Model Evaluation

The dataset is divided into training and testing data.

The model is trained using the training data and evaluated using previously unseen testing data.

The application displays the resulting test accuracy.

🌐 Live Demo

Open the deployed application

💻 How to Run Locally

Install the required libraries:

pip install -r requirements.txt


Run the application:

streamlit run app.py

🚀 Future Improvements

Possible improvements include:

Using a Convolutional Neural Network (CNN)

Allowing users to draw digits directly on the webpage

Improving image preprocessing

Comparing multiple machine learning algorithms

Adding confusion matrix visualization

Improving the user interface

👩‍💻 Author

YOUR NAME

This project was completed as part of an internship project.
