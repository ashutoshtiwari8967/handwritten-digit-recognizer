import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Page settings
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢"
)


# Train the machine learning model
@st.cache_resource
def train_model():

    digits = load_digits()

    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy


# Load model
model, accuracy = train_model()

digits = load_digits()


# Title
st.title("🔢 Handwritten Digit Recognizer")

st.write(
    """
    This application uses Machine Learning to recognize
    handwritten digits from 0 to 9.
    """
)


# Accuracy
st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)


st.divider()


# Select image
st.subheader("🖼️ Try the Model")

image_number = st.slider(
    "Select a handwritten digit",
    0,
    len(digits.images) - 1,
    0
)


# Get image
image = digits.images[image_number]

actual_digit = digits.target[image_number]


# Make prediction
prediction = model.predict(
    image.reshape(1, -1)
)[0]


# Create two columns
col1, col2 = st.columns(2)


with col1:

    st.write("### Input Image")

    fig, ax = plt.subplots(
        figsize=(3, 3)
    )

    ax.imshow(
        image,
        cmap="gray_r"
    )

    ax.axis("off")

    st.pyplot(fig)


with col2:

    st.write("### Prediction")

    st.write(
        f"**Actual Digit:** {actual_digit}"
    )

    st.write(
        f"**Predicted Digit:** {prediction}"
    )

    if prediction == actual_digit:

        st.success(
            "Correct prediction! ✅"
        )

    else:

        st.error(
            "The prediction is incorrect."
        )


st.divider()


# Explanation
st.subheader("📚 How It Works")

st.markdown(
    """
    **1. Dataset**

    The application uses the handwritten digits dataset
    provided by Scikit-learn.

    **2. Training**

    The data is divided into training and testing sets.

    **3. Machine Learning**

    A K-Nearest Neighbors (KNN) classifier learns patterns
    from the handwritten digits.

    **4. Prediction**

    The trained model predicts which digit an image represents.

    **5. Result**

    The application displays the actual digit,
    predicted digit, and model accuracy.
    """
)


st.info(
    "Built using Python, Scikit-learn, NumPy, Matplotlib and Streamlit."
)

