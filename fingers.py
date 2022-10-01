import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# define a  the camera object
cap = cv2.VideoCapture(0)

# define the fingers landmarks
tiplamd = [8, 12, 16, 20]



def draw_hand( image, hand_landmarks):
    '''this function draws the hand landmarks 
        Args:
            image (numpy.ndarray) : the frame comming from camera capture
            hand_landmarks(mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList) : the normalized hand landmarks
    '''
    return mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())


def count_fingers_up( fingers_up, hand_landmarks):
    '''Counting the remaining raised fingers except the thumb 
        Args :
            fingers_up (int): the number of raised thumbs 
            hand_landmarks(mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList) : the normalized hand andmarks
        Return : 
            fingers_up (int): the total number of raised fingers  

    '''
    for id in range(0, 4):
        if hand_landmarks.landmark[tiplamd[id]].y < hand_landmarks.landmark[tiplamd[id]-2].y:
            fingers_up += 1
    return fingers_up


def count_fingers( image, results):
    '''This function is responsable for counting the raised fingers in hands 
        Args : 
            image (numpy.ndarray) : the frame comming from camera capture 
            results (type) : the obtained output from the mediapipe model 
        Return :
             fingers_up (int) : the number of raised fingers in both hands 

    '''
    fingers_up = 0
    for label, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
        # right thumb finger translation
        if label.classification[0].index == 0: 
            if hand_landmarks.landmark[tiplamd[0]].x < hand_landmarks.landmark[tiplamd[-1]].x: # checking if the right hand is fliped
                # and the thumb is up
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    fingers_up += 1
            
            else: # else checking if the right hand is straight
                  # and the thumb is up
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    fingers_up += 1

                   
        else:  # else the left thumb finger translation

            if hand_landmarks.landmark[tiplamd[0]].x > hand_landmarks.landmark[tiplamd[-1]].x: # checking if the left hand is fliped
                # and the thumb is up
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    fingers_up += 1
            
            else: # else the left hand is straight
                # and the thumb is up
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    fingers_up += 1

        # counting the other fingers 
        fingers_up = count_fingers_up( fingers_up, hand_landmarks)
        # drawing the hand landmarks (the image passed by refrence) 
        draw_hand( image, hand_landmarks)

    return fingers_up


def post_process(image, fingers_up):
    '''this function convert the colores channels ,flip the image and write the result on it
        Args:
            image (numpy.ndarray) : the frame comming from the drawing
            fingers_up (int) : how many fingers up in the image 
        Return:
            image (numpy.ndarray) : the frame after post processing 
    '''

    # converting the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # flip the image 
    image = cv2.flip(image, 1)

    # putting the result on the image 
    cv2.putText(image, f'up fingers {fingers_up}', org=(
                50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)

    return image
def main():
    """ the main function"""

    # the model hypird parameters initialization
    with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            # reading the frame from the camera
            ret, image = cap.read()
            # checking of successful reading
            if ret:
                # for faster inference speed
                image.flags.writeable = False

                # converting the frame from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # predicting the landmarks for the hands from the image
                results = hands.process(image)

                # initiate the number of the appeared fingers with 0
                fingers_up = 0

                # checking if there any hands detected in the image
                if results.multi_hand_landmarks:
                    # looping over the appeared hands in the image
                    fingers_up = count_fingers(image, results)

                image = post_process(image, fingers_up)

                cv2.imshow('Fingers counter', image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

if __name__  == '__main__':
    main()
