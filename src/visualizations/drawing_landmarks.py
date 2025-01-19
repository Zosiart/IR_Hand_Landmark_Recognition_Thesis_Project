from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draws hand landmarks and handedness labels on an RGB image based on detection results
    from MediaPipe's GestureRecognizer.

    :param rgb_image: np.ndarray
        The input RGB image as a numpy array where hand landmarks will be drawn.
    :param detection_result: MediaPipe GestureRecognizer detection result object
        The detection result containing hand landmarks and handedness information.
    :return: np.ndarray
        The annotated RGB image with hand landmarks and handedness labels.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def draw_landmarks_on_image_from_json(rgb_image, evaluation_result):
    """
    Draws hand landmarks on an RGB image using ground truth landmarks stored in JSON format.

    :param rgb_image: np.ndarray
        The input RGB image as a numpy array where hand landmarks will be drawn.
    :param evaluation_result: dict
        A dictionary containing evaluation results, including hand landmarks.
        Example structure:
        {
            "landmarks": [
                [{"x": ..., "y": ...}, ...],  # Landmarks for hand 1
                [{"x": ..., "y": ...}, ...]   # Landmarks for hand 2
            ]
        }
    :return: np.ndarray
        The annotated RGB image with hand landmarks.
    """
    hand_landmarks_list = evaluation_result['landmarks']
    # handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        height, width, _ = annotated_image.shape
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=float(landmark['x']), y=float(landmark['y']), z=0.0) for landmark in
            hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        x_coordinates = [landmark['x'] for landmark in hand_landmarks]
        y_coordinates = [landmark['y'] for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(annotated_image, f"",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image