import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from tensorflow.keras.models import load_model
import math
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image

fontpath = "your font"
font = ImageFont.truetype(fontpath, 40)

actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
seq_length = 10

detector = hm.HolisticDetector(min_detection_confidence=0.3)

interpreter = tf.lite.Interpreter(model_path="your model")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = detector.findHolistic(img, draw=True)
    _, right_hand_lmList = detector.findRighthandLandmark(img)

    if right_hand_lmList is not None:
        joint = np.zeros((42, 2))

        for j, lm in enumerate(right_hand_lmList.landmark):
            joint[j] = [lm.x, lm.y]

        vector, angle_label = Vector_Normalization(joint)

        d = np.concatenate([vector.flatten(), angle_label.flatten()])

        seq.append(d)

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        input_data = np.array(input_data, dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        y_pred = interpreter.get_tensor(output_details[0]['index'])
        i_pred = int(np.argmax(y_pred[0]))
        conf = y_pred[0][i_pred]

        if conf < 0.9:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action
            if last_action != this_action:
                last_action = this_action

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), f'{action.upper()}', font=font, fill=(255, 255, 255))

        img = np.array(img_pil)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
