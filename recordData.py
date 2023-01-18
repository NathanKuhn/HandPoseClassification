import cv2
import mediapipe as mp
import pygame
import numpy as np
import pandas as pd
import math
import os

DATASET_PATH = 'training_dataset/'

def format_data(array):
    center_x = 0
    center_y = 0
    center_z = 0

    for i in range(array.shape[0]):
        center_x += array[i,0]
        center_y += array[i,1]
        center_z += array[i,2]
            
    center_x /= array.shape[0]
    center_y /= array.shape[0]
    center_z /= array.shape[0]

    # Center hands and find furthest point from center

    max_magnitude = 0

    for i in range(array.shape[0]):
        array[i, 0] = (array[i, 0] - center_x)
        array[i, 1] = (array[i, 1] - center_y)
        array[i, 2] = (array[i, 2] - center_z)

        max_magnitude = max(max_magnitude, math.hypot(array[i, 0], array[i, 1]))

    # Normalize points according to furthest point

    for i in range(array.shape[0]):
        array[i,0] = array[i, 0] / max_magnitude
        array[i,1] = array[i, 1] / max_magnitude
        array[i,2] = array[i, 2] / max_magnitude

    return array


def main():

    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 100)

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("camera")
    cv2.moveWindow("camera", 800, 100)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands()

    pygame.init()

    label = int(input("Label: "))

    screen = pygame.display.set_mode((500, 500))

    data_array = []

    done = False
    while not done:
        _, image = cap.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        screen.fill((0, 0, 0))

        if (results.multi_hand_landmarks):
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

            ratio = image.shape[1] / image.shape[0]

            array = np.zeros((len(hand.landmark), 3))

            for i in range(array.shape[0]):
                array[i,0] = hand.landmark[i].x * ratio
                array[i,1] = hand.landmark[i].y
                array[i,2] = hand.landmark[i].z

            array = format_data(array)

            for i in range(array.shape[0]):
                pygame.draw.circle(screen, (0, 255, 0), (int(array[i,0] * 100) + 250, int(array[i,1] * 100) + 250), 4) 

            if pygame.key.get_pressed()[pygame.K_SPACE]:
                array = array.flatten() # range -1 to 1 of dim (63,)
                data_array.append(list(array) + [label])
                print("Collected data point #", len(data_array))
        
        cv2.imshow("camera", cv2.flip(image, 1))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            done = True

        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

    data = pd.DataFrame(data_array)

    files = os.listdir(DATASET_PATH)
    file_num = len(files)

    data.to_csv(DATASET_PATH + f'{file_num}_label_{label}.csv')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()