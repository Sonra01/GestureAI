# import cv2
# import mediapipe as mp
# import pandas as pd
#
# # Initialize MediaPipe Hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils
#
# gesture_name = "BAD"  # Set your gesture name here
# samples = 200          # Number of samples to collect
# data = []
#
# # Start video capture
# cap = cv2.VideoCapture(0)
# print(f"Collecting data for gesture: '{gesture_name}'")
#
# while len(data) < samples:
#     success, img = cap.read()
#     if not success:
#         break
#
#     # Convert image to RGB (MediaPipe uses RGB)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             row = []
#             for lm in hand_landmarks.landmark:
#                 row.extend([lm.x, lm.y, lm.z])  # Add x, y, z of each landmark
#             row.append(gesture_name)
#             data.append(row)
#
#             # Draw landmarks on the image
#             mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#     # Show sample count on the screen
#     cv2.putText(img, f"Samples collected: {len(data)}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     cv2.imshow("Collecting", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# # Save collected data to CSV
# df = pd.DataFrame(data)
# df.to_csv("gesture_data.csv", mode='a', header=False, index=False)
# print("Data collection complete.")






# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
#
# # Load collected gesture data
# df = pd.read_csv("gesture_data.csv", header=None)
# X = df.iloc[:, :-1].values  # Feature columns (x, y, z for each of 21 landmarks)
# y = df.iloc[:, -1].values   # Labels (gesture names)
#
# # Encode labels into numbers
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
#
# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)
#
# # Build a neural network
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X.shape[1],)),
#     Dense(64, activation='relu'),
#     Dense(len(np.unique(y_encoded)), activation='softmax')  # Output layer
# ])
#
# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
#
# # Save the model and label encoder
# model.save("gesture_model.h5")
# np.save("gesture_labels.npy", le.classes_)
#
# print("Model trained and saved.")






import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model and gesture labels
model = load_model("gesture_model.h5")
labels = np.load("gesture_labels.npy", allow_pickle=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    prediction = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # Predict gesture from model
            result = model.predict(np.array([row]), verbose=0)
            class_id = np.argmax(result)
            prediction = labels[class_id]

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display prediction
    cv2.putText(img, f"Prediction: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()