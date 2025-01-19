import numpy as np
from mediapipe.framework.formats import landmark_pb2


class LandmarkMerger:
    """
    A class to merge and process hand landmarks from two pipelines.
    """

    def __init__(self, first_pipeline_results, second_pipeline_results):
        """
        Initialize the LandmarkMerger with results from two pipelines.

        :param first_pipeline_results: Results from the first pipeline (MediaPipe).
        :param second_pipeline_results: Results from the second pipeline (MediaPipe).
        """
        self.first_pipeline_landmarks = self.create_landmarks_from_results(first_pipeline_results)
        self.second_pipeline_landmarks = self.create_landmarks_from_results(second_pipeline_results)
        self.final_landmarks = None

    @staticmethod
    def is_right_hand(landmarks):
        """
        Determines if the hand landmarks belong to the right hand.

        :param landmarks: List of landmarks for a hand.
        :return: True if the hand is likely the right hand, False otherwise.
        """
        counter = sum(1 for landmark in landmarks if landmark.x > 0.5)
        return counter > 10

    def create_landmarks_from_results(self, results):
        """
        Extracts landmarks, gestures, and scores from the recognition results.

        :param results: MediaPipe results containing hand landmarks and gestures.
        :return: A dictionary with landmarks, gestures, and scores for each hand.
        """
        landmarks = results.hand_landmarks
        gestures = results.gestures
        final_landmarks = {}

        for i, hand_landmarks in enumerate(landmarks):
            hand_gestures = gestures[i]
            handedness = 'Right' if self.is_right_hand(hand_landmarks) else 'Left'

            final_landmarks[handedness] = {
                "landmarks": hand_landmarks,
                "gestures": hand_gestures[0].category_name,
                "score": hand_gestures[0].score
            }

        return final_landmarks

    def merge_landmarks(self):
        """
        Merges landmarks from the two pipelines to generate final landmarks.
        """
        landmarks_gestures_score = {"Left": [], "Right": []}

        # Add landmarks from the first pipeline
        for hand in ["Left", "Right"]:
            if hand in self.first_pipeline_landmarks:
                landmarks_gestures_score[hand].append(self.first_pipeline_landmarks[hand])

        # Add landmarks from the second pipeline
        for hand in ["Left", "Right"]:
            if hand in self.second_pipeline_landmarks:
                landmarks_gestures_score[hand].append(self.second_pipeline_landmarks[hand])

        self.final_landmarks = []
        for hand in ["Left", "Right"]:
            self.process_hand_landmarks(landmarks_gestures_score[hand], hand)

    def process_hand_landmarks(self, landmarks_list, hand):
        """
        Processes landmarks for a specific hand by merging or selecting the best landmarks.

        :param landmarks_list: List of landmarks and their corresponding metadata.
        """
        if not landmarks_list:
            return

        open_palm_landmarks = [l for l in landmarks_list if l['gestures'] == 'Open_Palm']

        if len(open_palm_landmarks) > 1:
            merged_landmarks = self.merge_open_palm_landmarks(open_palm_landmarks, hand)
            self.final_landmarks.append(merged_landmarks)

        elif len(open_palm_landmarks) == 1:
            self.final_landmarks.append(open_palm_landmarks[0]['landmarks'])

        else:
            lowest_score_landmark = min(landmarks_list, key=lambda l: l['score'])
            self.final_landmarks.append(lowest_score_landmark['landmarks'])

    @staticmethod
    def merge_open_palm_landmarks(pipelines, handedness):
        """
        Merges multiple open palm landmarks for a specific hand.

        :param landmarks_list: List of open palm landmarks.
        :return: Merged landmarks for the hand.
        """

        def is_fingers_correct_order(landmarks):
            # Extract x-coordinates for all landmarks
            x_coords = [landmarks[i].x for i in range(len(landmarks))]

            if handedness == "Right":
                # Right hand: all pinky points (6 to 20) should be to the left of the thumb (2, 3, 4)
                return all(x_coords[pinky] < x_coords[2] for pinky in range(10, 21))
            else:  # Left hand
                # Left hand: all pinky points (6 to 20) should be to the right of the thumb (2, 3, 4)
                return all(x_coords[pinky] > x_coords[2] for pinky in range(10, 21))

        def merge_logic(points, index):
            if index < 6:  # For the first 6 landmarks
                if handedness == "Left":
                    return min(points, key=lambda p: p.x)  # Leftmost for left hand
                else:
                    return max(points, key=lambda p: p.x)  # Rightmost for right hand
            else:  # For the remaining landmarks
                return min(points, key=lambda p: p.y)  # Lowest y

        correct_pipelines = [pipeline for pipeline in pipelines if is_fingers_correct_order(pipeline['landmarks'])]

        # If at least one correct pipeline is found
        if correct_pipelines:
            merged_landmarks = []
            num_landmarks = len(correct_pipelines[0]['landmarks'])

            for i in range(num_landmarks):
                # Extract corresponding landmarks from correct pipelines
                points = [pipeline['landmarks'][i] for pipeline in correct_pipelines]
                # Apply merging logic
                merged_landmarks.append(merge_logic(points, i))

            return merged_landmarks

        # No correct pipeline found, fall back to merging all pipelines
        merged_landmarks = []
        num_landmarks = len(pipelines[0]['landmarks'])

        for i in range(num_landmarks):
            points = [pipeline['landmarks'][i] for pipeline in pipelines]
            merged_landmarks.append(merge_logic(points, i))

        return merged_landmarks
