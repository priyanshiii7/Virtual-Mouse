import cv2
import mediapipe as mp
import utility
import pyautogui
import random
from pynput.mouse import Button, Controller
mouse = Controller()


screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mpHands.Hands(
  static_image_mode=False,    #becoz we're capturing vdo not a pic
  model_complexity=1,   #gets better model
  min_detection_confidence=0.6,  #min confidence score a hand requires to be detected if its too much the pointer will end up lagging (0.8)
  min_tracking_confidence=0.6,   
  max_num_hands=1 #only 1 hand is detected for a  mouse
)


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    
    return None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x,y)


def is_left_click(landmarks_list, thumb_index_dist):
    #check if index finger is bent, middle finger is straight and thumb is straight
    return (utility.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and 
            utility.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90 and
            thumb_index_dist > 50
            )


def is_right_click(landmarks_list, thumb_index_dist):
    #check if middle finger is bent, index finger is straight and thumb is straight
    return (utility.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and 
            utility.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90 and
            thumb_index_dist > 50
            )

def is_double_click(landmarks_list, thumb_index_dist): 
    return (utility.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and 
            utility.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
            thumb_index_dist > 50
            )

def is_screenshot(landmarks_list, thumb_index_dist):
    #check for fist
    return (utility.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and 
            utility.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
            thumb_index_dist < 50
            )


def detect_gestures(frame, landmarks_list, processed):
    if len(landmarks_list)>=21:
        #choosing tip as pointer of mouse
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = utility.get_distance([landmarks_list[4], landmarks_list[5]])

        if thumb_index_dist < 50 and utility.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90:
            move_mouse(index_finger_tip)


        #left click is defined here w conditions
        elif is_left_click(landmarks_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)

        #right click
        elif is_right_click(landmarks_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2)

        #double click
        elif is_double_click(landmarks_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255, 0), 2)

        #ss
        #fist
        elif is_screenshot(landmarks_list, thumb_index_dist):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)
         

def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            frame = cv2.flip(frame, 1)  #to mirror the frame
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmarks_list = []

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))   #prints coordinates of hand landmarks
                    
            #code to detect different hand gestures
            detect_gestures(frame, landmarks_list, processed)

        

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
             break    #tells opencv to break the frame when q is clicked.
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 