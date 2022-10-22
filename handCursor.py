import json
import time
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import mouse

from recordData import format_data
from trainModel import create_model

def map_to_screen(x, y):
    x = 1 - x

    x *= 1920 * 1.3
    y *= 1080 * 1.3

    x -= 300
    y -= 300

    return (x, y)

def main():

    with open('label_names.json', 'r') as json_file:
        labels = json.load(json_file)
    
    label_dict = {v : k for k, v in labels.items()}

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands()

    model = create_model()
    model.load_weights("weights/checkpoint")

    mp_time = 0
    model_time = 0

    done = False
    while not done:
        _, image = cap.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        timer = time.time()
        results = hands.process(image_rgb)
        mp_time = time.time() - timer

        result = ""

        if (results.multi_hand_landmarks):
            hand = results.multi_hand_landmarks[0]
            #mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

            ratio = image.shape[1] / image.shape[0]

            array = np.zeros((len(hand.landmark), 3))

            for i in range(array.shape[0]):
                array[i,0] = hand.landmark[i].x * ratio
                array[i,1] = hand.landmark[i].y
                array[i,2] = hand.landmark[i].z

            hand_x = 0
            hand_y = 0

            for i in [0, 5, 17]:
                hand_x += hand.landmark[i].x
                hand_y += hand.landmark[i].y

            hand_x /= 3
            hand_y /= 3

            array = format_data(array)
            array = array.flatten()

            array = np.array([array])

            timer = time.time()
            prediction = model.predict(array, verbose=0)[0]
            model_time = time.time() - timer

            max_index = np.argmax(prediction)
            result = label_dict[max_index]

            cv2.circle(image, (int(hand_x * image.shape[0]), int(hand_y * image.shape[0])), 5, (255, 10, 10), 3)

            if (result in ["5", "fist", "vulcan"]):
                mouse.move(*map_to_screen(hand_x, hand_y))

            if (result == "fist"):
                mouse.press()
            else:
                mouse.release()

        image = cv2.flip(image, 1)

        cv2.imshow("display", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            done = True
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()