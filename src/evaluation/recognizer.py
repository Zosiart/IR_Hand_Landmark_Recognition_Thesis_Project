from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

class Recognizer:

    def __init__(self, model_path):
        """
        Initialize the Recognizer object with the path to the model.

        :param model_path (str): The path to the model file.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(base_options=base_options,
                                                  num_hands=2)

        options.canned_gesture_classifier_options.score_threshold = 0
        self.recognizer = vision.GestureRecognizer.create_from_options(options)


    def recognize_landmarks_gestures(self, image_path):
        """
        Recognize the landmarks in the input image.

        :param image_path: The path to the input image.
        :return: The recognized landmarks.
        """
        image = mp.Image.create_from_file(image_path)
        results = self.recognizer.recognize(image)

        return results
