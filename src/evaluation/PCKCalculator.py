import numpy as np


class PCKCalculator:
    """
    Class to calculate the Percentage of Correct Keypoints (PCK) metric.
    """


    def __init__(self, threshold=0.05):
        """
        Initialize the PCKCalculator object with distance threshold factor.

        :param threshold (float): The distance threshold factor. Used to calculate acceptable distance from ground
         truth point.
        """
        self.threshold = threshold

    def set_threshold(self, threshold):
        """
        Set a new distance threshold factor.

        :param threshold: The distance threshold factor. Used to calculate acceptable distance from ground
         truth point.
        """
        self.threshold = threshold

    def calculate_pck(self, ground_truth_hands, predicted_hands):
        """
        Calculate the Percentage of Correct Keypoints (PCK) for the predicted landmarks.

        :param ground_truth_hands: A list containing list of landmarks for each hand in the image.
        :param predicted_hands: A list containing list of landmarks for each hand in the image.
        """

        if not predicted_hands or len((predicted_hands[0])) == 0:
            return {"Left": 0.00, "Right": 0.00}

        return self.calculate_best_pck_combination(ground_truth_hands, predicted_hands)

    def calculate_acceptable_distance(self, ground_truth_hand):
        """
        Calculate an acceptable distance based on the distance between key landmarks.

        :param ground_truth_hand: The ground truth landmarks for a hand.
        :return: The acceptable distance.
        """
        wrist_point = ground_truth_hand[0]
        top_of_middle_finger = ground_truth_hand[12]

        distance = np.sqrt((wrist_point['x'] - top_of_middle_finger['x']) ** 2 +
                           (wrist_point['y'] - top_of_middle_finger['y']) ** 2)
        return distance * self.threshold


    @staticmethod
    def calculate_pck_for_hand(ground_truth_hand, predicted_hand, acceptable_distance):
        """
        Calculate the PCK metric for a single hand.

        :param ground_truth_hand: Ground truth landmarks for a hand.
        :param predicted_hand: Predicted landmarks for a hand.
        :param acceptable_distance: The acceptable distance for the landmark.
        :return: The PCK score for the hand.
        """
        correct_landmarks = 0

        for gt_landmark, pred_landmark in zip(ground_truth_hand, predicted_hand):
            distance = np.sqrt((gt_landmark['x'] - pred_landmark.x) ** 2 +
                               (gt_landmark['y'] - pred_landmark.y) ** 2)
            if distance <= acceptable_distance:
                correct_landmarks += 1

        return correct_landmarks / len(ground_truth_hand)

    def calculate_best_pck_combination(self, ground_truth_hands, predicted_hands):
        """
        Calculate the best PCK combination for multiple hands.

        :param ground_truth_hands: A list of ground truth landmarks for both hands.
        :param predicted_hands: A list of predicted landmarks for both hands.
        :return: A dictionary with PCK scores for left and right hands.
        """
        pck = {"Left": 0.0, "Right": 0.0}

        first_predicted_hand = predicted_hands[0]
        acceptable_distance_left = self.calculate_acceptable_distance(ground_truth_hands[0])
        acceptable_distance_right = self.calculate_acceptable_distance(ground_truth_hands[1])

        left_pck_first = self.calculate_pck_for_hand(ground_truth_hands[0], first_predicted_hand, acceptable_distance_left)
        right_pck_first = self.calculate_pck_for_hand(ground_truth_hands[1], first_predicted_hand, acceptable_distance_right)

        if len(predicted_hands) == 1:
            if left_pck_first > right_pck_first:
                pck["Left"] = left_pck_first
            else:
                pck["Right"] = right_pck_first
            return pck

        second_predicted_hand = predicted_hands[1]
        left_pck_second = self.calculate_pck_for_hand(ground_truth_hands[0], second_predicted_hand, acceptable_distance_left)
        right_pck_second = self.calculate_pck_for_hand(ground_truth_hands[1], second_predicted_hand, acceptable_distance_right)

        if left_pck_first + right_pck_second > right_pck_first + left_pck_second:
            pck["Left"] = left_pck_first
            pck["Right"] = right_pck_second
        else:
            pck["Left"] = left_pck_second
            pck["Right"] = right_pck_first

        return pck

