import json
import time
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

from recordData import format_data
from trainModel import create_model

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
        text_pos_x = 0
        text_pos_y = 0

        if (results.multi_hand_landmarks):
            hand = results.multi_hand_landmarks[0]
            #mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

            ratio = image.shape[1] / image.shape[0]

            array = np.zeros((len(hand.landmark), 3))

            min_x = 1
            min_y = ratio
            max_x = 0
            max_y = 0

            for i in range(array.shape[0]):
                array[i,0] = hand.landmark[i].x * ratio
                array[i,1] = hand.landmark[i].y
                array[i,2] = hand.landmark[i].z

                max_x = max(max_x, array[i,0])
                min_x = min(min_x, array[i,0])
                max_y = max(max_y, array[i,1])
                min_y = min(min_y, array[i,1])

            array = format_data(array)
            array = array.flatten()

            array = np.array([array])

            timer = time.time()
            prediction = model.predict(array, verbose=0)[0]
            model_time = time.time() - timer

            max_index = np.argmax(prediction)

            cv2.rectangle(image, (int(min_x * image.shape[0]), int(min_y * image.shape[0])), (int(max_x * image.shape[0]), int(max_y * image.shape[0])), (255, 10, 10), 2)
            text_pos_x = int((1 - (min_x + max_x) * 0.5) * image.shape[0]) + 150
            text_pos_y = int(max_y * image.shape[0] + 20)
            result = label_dict[max_index]

        else:
            model_time = 0

        img = cv2.flip(image, 1)
        cv2.putText(img, f'Media Pipe: {mp_time*1000:>5.2f} ms', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 30, 30))
        cv2.putText(img, f'Pose Model: {model_time*1000:>5.2f} ms', (10, 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 30, 30))
        cv2.putText(img, result, (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (30, 30, 255))

        cv2.imshow("display", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            done = True
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()